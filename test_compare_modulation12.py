import unittest
from compare_modulation12 import summarize

class ComparisonTests(unittest.TestCase):
    def test_aggregate_keeps_architectures_separate_and_reports_sample_spread(self):
        runs=[]
        for architecture,values in [('legacy',[.4,.6]),('harper2023',[.6,.8])]:
            for accuracy in values:
                runs.append(dict(architecture=architecture,test_accuracy=accuracy,macro_f1=accuracy,
                    recall={'fm':accuracy},by_snr={'10.0':{'accuracy':accuracy}},
                    frequency_tests={str(h):{'accuracy':accuracy-.1} for h in (-1600,1600)}))
        result=summarize(runs)
        self.assertAlmostEqual(result['legacy']['mean_accuracy'],.5)
        self.assertAlmostEqual(result['harper2023']['mean_accuracy'],.7)
        self.assertAlmostEqual(result['legacy']['sample_std_accuracy'],.02**.5)
        self.assertAlmostEqual(result['harper2023']['mean_recall']['fm'],.7)
        self.assertAlmostEqual(result['harper2023']['mean_accuracy_by_snr']['10.0'],.7)
        self.assertAlmostEqual(result['harper2023']['mean_frequency_accuracy']['1600'],.6)

if __name__=='__main__': unittest.main()
