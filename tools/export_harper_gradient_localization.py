"""Original TensorFlow gradient decomposition; Python 3.8 / TensorFlow 2.4.1."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import numpy as np
import tensorflow as tf
from export_harper_tf241 import SOURCE,verify_source,key


def export(fixtures,output):
    if tf.__version__!='2.4.1' or np.__version__!='1.19.2': raise ValueError('Wrong reference environment')
    verify_source()
    sys.path[:0]=[str(SOURCE/'experiments/radioml2018.01a'),str(SOURCE)]
    import models
    fixtures=Path(fixtures);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    original=json.loads((fixtures/'export.json').read_text());specs=[]
    for spec in original['cases']:
        path=fixtures/(spec['name']+'.npz')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']: raise ValueError('Fixture changed')
        with np.load(path,allow_pickle=False) as saved: source={k:saved[k] for k in saved.files}
        for dtype in ['float32','float64']:
            tf.keras.backend.clear_session();tf.keras.backend.set_floatx(dtype)
            tf.keras.mixed_precision.set_global_policy(dtype)
            model=models.create_model(input_shape=(spec['samples'],2),n_classes=spec['classes'])
            if model.weights[0].dtype.name!=dtype: raise ValueError('Unexpected model dtype')
            for var in model.trainable_variables: var.assign(source['weight__'+key(var.name)].astype(dtype))
            names=['conv_'+str(i) for i in range(1,8)]+['act_variant_2']
            probe=tf.keras.Model(model.inputs,[model.get_layer(n).output for n in names])
            last=model.get_layer('variant');x=source['input'].astype(dtype);labels=source['labels']
            with tf.GradientTape(persistent=True) as tape:
                values=probe(x,training=False)
                logits=tf.matmul(values[-1],last.kernel)+last.bias
                losses=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
                loss=tf.reduce_mean(losses)
                per_row=[losses[i]/3 for i in range(3)]
            arrays=dict(input=x,labels=labels,logits=logits.numpy(),loss=np.array(loss.numpy()))
            full=tape.gradient(loss,model.trainable_variables)
            for var,g in zip(model.trainable_variables,full):
                if g is None: raise ValueError('Missing full gradient')
                arrays['gfull__'+key(var.name)]=g.numpy()
                arrays['weight__'+key(var.name)]=var.numpy()
            for row in range(3):
                grads=tape.gradient(per_row[row],model.trainable_variables)
                for var,g in zip(model.trainable_variables,grads): arrays['grow'+str(row)+'__'+key(var.name)]=g.numpy()
            for i,name in enumerate(names[:-1]): arrays['activation__'+name]=values[i].numpy()
            for layer in (5,7):
                act=values[layer-1];u=tape.gradient(loss,act)
                if u is None: raise ValueError('Missing cotangent')
                q=u*tf.cast(act>0,act.dtype)
                arrays['cotangent__conv_'+str(layer)]=u.numpy()
                arrays['preactivation_cotangent__conv_'+str(layer)]=q.numpy()
                bias=tf.Variable(tf.zeros([int(q.shape[-1])],dtype=q.dtype))
                with tf.GradientTape() as isolated:
                    result=tf.nn.bias_add(tf.zeros_like(q),bias)
                    objective=tf.reduce_sum(result*tf.stop_gradient(q))
                arrays['bias_vjp__conv_'+str(layer)]=isolated.gradient(objective,bias).numpy()
            del tape
            name=spec['name']+'-'+dtype
            np.savez_compressed(output/(name+'.npz'),**arrays)
            specs.append(dict(**spec,dtype=dtype,diagnostic_name=name,
                diagnostic_sha256=hashlib.sha256((output/(name+'.npz')).read_bytes()).hexdigest()))
            print('TF gradient decomposition:',name,flush=True)
    report=dict(schema=1,plan='GRADIENT_LOCALIZATION_PLAN.json',tensorflow=tf.__version__,numpy=np.__version__,
                cases=specs,reference_values='original float32 weights and inputs reused; exact promotion to float64',
                source_manifest_sha256=hashlib.sha256((SOURCE/'SOURCE_MANIFEST.json').read_bytes()).hexdigest())
    (output/'localization_export.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();export(args.fixtures,args.output)
