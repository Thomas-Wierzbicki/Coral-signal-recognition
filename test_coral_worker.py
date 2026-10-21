"""Protokoll- und Quantisierungstests ohne Coral-Hardware."""
import base64
from contextlib import redirect_stdout
import io
import json
import unittest
import numpy as np
from coral_worker import CoralClassifier, decode_iq, serve
from signal_lab import SAMPLES


class FakeInterpreter:
    def __init__(self):
        self.calls = 0
    def allocate_tensors(self):
        pass
    def get_input_details(self):
        return [dict(index=0, shape=(1,64,256,1), dtype=np.uint8, quantization=(1/255,0))]
    def get_output_details(self):
        return [dict(index=1, shape=(1,3), dtype=np.uint8, quantization=(1/256,0))]
    def set_tensor(self, index, tensor):
        assert tensor.shape == (1,64,256,1) and tensor.dtype == np.uint8
    def invoke(self):
        self.calls += 1
    def get_tensor(self, index):
        return np.array([[16,32,208]], dtype=np.uint8)


def request():
    iq = np.ones(SAMPLES, dtype="<c8") * (0.25+0.5j)
    return dict(type="classify", id=7, sample_rate=240000,
                iq_base64=base64.b64encode(iq.tobytes()).decode("ascii"))


class WorkerTests(unittest.TestCase):
    def test_iq_roundtrip_and_rejections(self):
        r = request()
        np.testing.assert_equal(decode_iq(r), np.full(SAMPLES,0.25+0.5j,np.complex64))
        for change in (dict(sample_rate=2400000),dict(iq_base64="!!!"),
                       dict(iq_base64="AAAA"),
                       dict(iq_base64=base64.b64encode(np.full(SAMPLES,np.nan,dtype="<c8").tobytes()).decode())):
            with self.assertRaises(ValueError):
                decode_iq(dict(r, **change))
    def test_single_interpreter_reused_and_error_recovery(self):
        interpreter = FakeInterpreter()
        created = []
        def factory(path):
            created.append(path)
            return interpreter
        classifier = CoralClassifier("fake", factory=factory)
        output = io.StringIO()
        messages = ["[]",json.dumps(dict(request(),sample_rate=1)),
                    json.dumps(request()),json.dumps(dict(request(),id=8)),
                    json.dumps(dict(type="shutdown",id=9))]
        with redirect_stdout(output):
            code = serve(classifier, io.StringIO("\n".join(messages)+"\n"))
        responses = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(code,0)
        self.assertEqual([r["type"] for r in responses],
                         ["ready","error","error","result","result","bye"])
        self.assertEqual(responses[3]["label"],"fm")
        self.assertEqual(responses[3]["scores"]["fm"],208/256)
        self.assertEqual(responses[3]["id"],7)
        self.assertEqual(len(created),1)
        self.assertEqual(interpreter.calls,3)  # Ein Aufwaermlauf, zwei Auftraege.


if __name__ == "__main__":
    unittest.main()
