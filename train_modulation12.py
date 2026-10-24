"""GPU-Referenztraining fuer modulation12-v1; Auswahl nach Validierung, Test erst danach."""
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from build_modulation_dataset import sha
from torchsig_probe import CLASSES,RATE,SAMPLES


def load_metadata(folder):
    folder=Path(folder)
    if not (folder/'DATASET_COMPLETE').is_file(): raise ValueError('DATASET_COMPLETE fehlt')
    meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    if (meta.get('dataset_id')!='modulation12-controlled-fm-v1' or meta.get('classes')!=CLASSES
            or meta.get('sample_rate_hz')!=RATE or meta.get('samples_per_example')!=SAMPLES
            or meta.get('normalization')!='block-rms-0.5'):
        raise ValueError('Datensatzmetadaten passen nicht')
    return meta


def load_split(folder,name,metadata,seen):
    path=Path(folder)/(name+'.npz'); spec=metadata['splits'][name]
    if sha(path)!=spec['sha256']: raise ValueError('NPZ-Hash stimmt nicht: '+name)
    with np.load(path,allow_pickle=False) as d:
        iq,y,ids,snr=(d[k] for k in ('iq','y','source_id','snr_db'))
    if (iq.shape!=(len(y),SAMPLES) or len(y)!=spec['examples'] or ids.shape!=y.shape or snr.shape!=y.shape
            or not np.iscomplexobj(iq) or not np.isfinite(iq).all()
            or not np.issubdtype(y.dtype,np.integer) or set(y.tolist())!=set(range(len(CLASSES)))
            or not np.all(np.bincount(y,minlength=len(CLASSES))==spec['per_class'])
            or not np.isfinite(snr[y!=CLASSES.index('noise')]).all()
            or not np.isnan(snr[y==CLASSES.index('noise')]).all()):
        raise ValueError('Ungueltige Arrays: '+name)
    np.testing.assert_allclose(np.sqrt(np.mean(abs(iq)**2,axis=1)),.5,atol=1e-5)
    unique=set(ids.tolist())
    if len(unique)!=len(ids) or unique&seen: raise ValueError('Quell-IDs ueberlappen: '+name)
    seen.update(unique)
    return iq.astype(np.complex64),y.astype(np.int64),ids,snr


def as_tensor(iq):
    # NCHW: zwei Kanaele, eine Hoehenzeile, 4096 Zeitpunkte.
    return torch.from_numpy(np.stack((iq.real,iq.imag),axis=1).astype(np.float32))[:,:,None,:]


def make_model(architecture='legacy'):
    if architecture=='harper2023':
        from modulation_reference import Harper2023
        return Harper2023(len(CLASSES))
    if architecture!='legacy': raise ValueError('Unbekannte Architektur: '+architecture)
    return nn.Sequential(
        nn.Conv2d(2,24,(1,9),stride=(1,2),padding=(0,4)),nn.ReLU(),nn.MaxPool2d((1,2)),
        nn.Conv2d(24,48,(1,7),stride=(1,2),padding=(0,3)),nn.ReLU(),nn.MaxPool2d((1,2)),
        nn.Conv2d(48,64,(1,5),stride=(1,2),padding=(0,2)),nn.ReLU(),nn.MaxPool2d((1,4)),
        nn.Flatten(),nn.Linear(2048,128),nn.ReLU(),nn.Dropout(.2),nn.Linear(128,len(CLASSES)))


def metrics(y,pred):
    matrix=np.zeros((len(CLASSES),len(CLASSES)),dtype=np.int64)
    np.add.at(matrix,(y,pred),1)
    correct=np.diag(matrix).astype(float)
    recall=np.divide(correct,matrix.sum(axis=1),out=np.zeros_like(correct),where=matrix.sum(axis=1)>0)
    precision=np.divide(correct,matrix.sum(axis=0),out=np.zeros_like(correct),where=matrix.sum(axis=0)>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros_like(correct),where=precision+recall>0)
    return dict(count=len(y),accuracy=float(np.mean(y==pred)),macro_f1=float(f1.mean()),
        confusion_matrix=matrix.tolist(),recall=dict(zip(CLASSES,recall.tolist())),
        precision=dict(zip(CLASSES,precision.tolist())))


def predict(model,inputs,device,batch_size):
    model.eval(); predictions=[]
    with torch.inference_mode():
        for (batch,) in DataLoader(TensorDataset(inputs),batch_size=batch_size,shuffle=False):
            predictions.append(model(batch.to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def shifted(iq,frequency_hz):
    return (iq*np.exp(2j*np.pi*frequency_hz/RATE*np.arange(SAMPLES))).astype(np.complex64)


def train(data,output,epochs=40,batch_size=32,seed=2051,device_name='cuda',patience=8,architecture='legacy'):
    data=Path(data); output=Path(output)
    if output.exists(): raise ValueError('Ausgabeordner existiert bereits; neuen --output verwenden')
    if epochs<1 or batch_size<1 or patience<1: raise ValueError('Positive Trainingsparameter erforderlich')
    if device_name=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA nicht verfuegbar; CPU nur explizit mit --device cpu')
    device=torch.device(device_name); metadata=load_metadata(data); seen=set()
    train_iq,train_y,_,_=load_split(data,'train',metadata,seen)
    val_iq,val_y,_,_=load_split(data,'validation',metadata,seen)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    loader=DataLoader(TensorDataset(as_tensor(train_iq),torch.from_numpy(train_y)),
        batch_size=batch_size,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(seed))
    val_loader=DataLoader(TensorDataset(as_tensor(val_iq),torch.from_numpy(val_y)),batch_size=batch_size,shuffle=False)
    model=make_model(architecture).to(device); optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    criterion=nn.CrossEntropyLoss(); output.mkdir(parents=True)
    config=dict(seed=seed,device=str(device),torch_version=torch.__version__,epochs=epochs,batch_size=batch_size,
        patience=patience,selection='minimum validation cross entropy',classes=CLASSES,
        architecture=architecture,
        model=('three Conv2D/ReLU/pooling stages, Dense128, Dropout0.2, 12 logits' if architecture=='legacy'
               else 'Harper2023 default architecture port; see reference provenance'),
        parameters=sum(p.numel() for p in model.parameters()),training_augmentation='none beyond generated dataset')
    if architecture=='harper2023':
        from modulation_reference import REFERENCE
        config['reference']=REFERENCE
    config['dataset_metadata_sha256']=sha(data/'metadata.json')
    config['optimizer']=dict(name='Adam',lr=.001,eps=1e-8,weight_decay=0)
    (output/'training_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('Geraet:',torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU',flush=True)
    print('Architektur:',architecture,'Modellparameter:',config['parameters'],flush=True)
    history=[]; best=float('inf'); best_epoch=0; bad=0
    checkpoint=output/'best_state_dict.pt'
    for epoch in range(1,epochs+1):
        start=time.perf_counter(); model.train(); total=0.; correct=0
        for batch,target in loader:
            batch=batch.to(device); target=target.to(device)
            optimizer.zero_grad(set_to_none=True); logits=model(batch); loss=criterion(logits,target)
            loss.backward(); optimizer.step()
            total+=float(loss.detach())*len(target); correct+=int((logits.argmax(1)==target).sum())
        model.eval(); val_loss=0.; val_correct=0
        with torch.inference_mode():
            for batch,target in val_loader:
                batch=batch.to(device); target=target.to(device); logits=model(batch)
                val_loss+=float(criterion(logits,target))*len(target)
                val_correct+=int((logits.argmax(1)==target).sum())
        val_loss/=len(val_y)
        row=dict(epoch=epoch,loss=total/len(train_y),accuracy=correct/len(train_y),
            val_loss=val_loss,val_accuracy=val_correct/len(val_y),seconds=time.perf_counter()-start)
        history.append(row)
        print('Epoche {epoch}: loss {loss:.4f}, accuracy {accuracy:.1%}, val_loss {val_loss:.4f}, val_accuracy {val_accuracy:.1%}, {seconds:.1f}s'.format(**row),flush=True)
        if not np.isfinite(row['loss']) or not np.isfinite(val_loss): raise ValueError('Nichtendlicher Trainingsverlust')
        if val_loss<best:
            best=val_loss; best_epoch=epoch; bad=0
            torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},checkpoint)
        else: bad+=1
        (output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        if bad>=patience: break
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)); model.eval()
    # Erst nach abgeschlossener Modellwahl den Testsatz laden und auswerten.
    test_iq,test_y,test_ids,test_snr=load_split(data,'test',metadata,seen)
    base_inputs=as_tensor(test_iq); base_pred=predict(model,base_inputs,device,batch_size)
    result=metrics(test_y,base_pred)
    report=dict(config=config,dataset_metadata=metadata,best_epoch=best_epoch,test=result,
        by_snr={str(float(s)):metrics(test_y[test_snr==s],base_pred[test_snr==s]) for s in np.unique(test_snr[np.isfinite(test_snr)])},
        noise=metrics(test_y[np.isnan(test_snr)],base_pred[np.isnan(test_snr)]),frequency_tests={})
    saved=dict(true=test_y,source_id=test_ids,snr_db=test_snr,original=base_pred)
    for hz in (-1600,1600):
        pred=predict(model,as_tensor(shifted(test_iq,hz)),device,batch_size)
        stat=metrics(test_y,pred); stat['agreement_with_original']=float(np.mean(base_pred==pred))
        report['frequency_tests'][str(hz)]=stat; saved['shift_'+str(hz)]=pred
    np.savez_compressed(output/'test_predictions.npz',**saved)
    # Portabler Float-Inferenzexport, noch keine TFLite-/Coral-Konvertierung.
    model=model.cpu().eval()
    example=base_inputs[:min(4,len(base_inputs))]
    traced=torch.jit.trace(model,torch.zeros(1,2,1,SAMPLES)); traced.save(str(output/'model_traced.pt'))
    loaded=torch.jit.load(str(output/'model_traced.pt')).eval()
    with torch.inference_mode(): torch.testing.assert_close(model(example),loaded(example))
    report.update(state_dict_sha256=sha(checkpoint),traced_model_sha256=sha(output/'model_traced.pt'),
        limitations='Synthetic same-generator split; not real-world or drone validation. Fixed shifted tests are paired development checks, not independent samples. No unknown rejection. No Coral export yet.')
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    manifest=dict(schema=1,task='modulation12',classes=CLASSES,input_shape=[1,2,1,SAMPLES],
        input_layout='NCHW: I,Q channels; one height row; time samples',sample_rate_hz=RATE,
        normalization='complex block RMS 0.5',representation='raw_iq',output='12 uncalibrated logits',
        source_dataset_id=metadata['dataset_id'],model_sha256=report['traced_model_sha256'])
    (output/'preprocessing.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    plot_matrix(output,np.array(result['confusion_matrix']))
    (output/'TRAINING_COMPLETE').write_text('OK\n')
    print('\nBeste Epoche:',best_epoch)
    print('Synthetische Testgenauigkeit: {:.1%}; Macro-F1: {:.3f}'.format(result['accuracy'],result['macro_f1']))
    print('Klassenfolge:',CLASSES); print('Zeilen Soll, Spalten Vorhersage:'); print(np.asarray(result['confusion_matrix']))
    for label,value in result['recall'].items(): print('{}: {:.1%}'.format(label,value))
    for hz,stat in report['frequency_tests'].items():
        print('Verschiebung {} Hz: {:.1%}; Uebereinstimmung {:.1%}'.format(hz,stat['accuracy'],stat['agreement_with_original']))
    print('Export:',output.resolve()); print('Float-Modell; noch kein Coral- oder Realempfangstest.')
    return report


def plot_matrix(output,matrix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,9),layout='constrained'); ax.imshow(matrix,cmap='Blues')
    ax.set_xticks(range(len(CLASSES)),CLASSES,rotation=60,ha='right'); ax.set_yticks(range(len(CLASSES)),CLASSES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)): ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=8)
    fig.savefig(output/'confusion_matrix.png',dpi=140); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/modulation12-v1'))
    p.add_argument('--output',type=Path,default=Path('models/modulation12-v1'))
    p.add_argument('--epochs',type=int,default=40); p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--seed',type=int,default=2051); p.add_argument('--patience',type=int,default=8)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--architecture',choices=['legacy','harper2023'],default='legacy'); args=p.parse_args()
    try: train(args.data,args.output,args.epochs,args.batch_size,args.seed,args.device,args.patience,args.architecture)
    except (OSError,ValueError,KeyError,RuntimeError,AssertionError) as error:
        p.exit(1,'Referenztraining fehlgeschlagen: '+str(error)+'\n')


if __name__=='__main__': main()
