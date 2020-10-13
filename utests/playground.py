import pandas as pd
import previsionio as pio
from collections import namedtuple
import logging

pio.logger.setLevel(logging.DEBUG)
EventTuple = namedtuple('EventTuple', 'key value')

if __name__ == '__main__':
    pio.verbose(True, debug=True, event_log=True)
    cs = ['smart_9_normalized',
          'smart_9_raw', 'smart_10_normalized', 'smart_10_raw',
          'smart_11_normalized', 'smart_11_raw', 'smart_12_normalized',
          'smart_12_raw', 'smart_13_normalized', 'smart_13_raw',
          'smart_15_normalized', 'smart_15_raw', 'smart_22_normalized',
          'smart_22_raw', 'smart_183_normalized', 'smart_183_raw',
          'smart_184_normalized', 'smart_184_raw', 'smart_187_normalized',
          'smart_187_raw', 'smart_188_normalized', 'smart_188_raw',
          'smart_189_normalized', 'smart_189_raw', 'smart_190_normalized',
          'smart_190_raw', 'smart_191_normalized', 'smart_191_raw',
          'smart_192_normalized', 'smart_192_raw', 'smart_193_normalized',
          'smart_193_raw', 'smart_194_normalized', 'smart_194_raw',
          'smart_195_normalized', 'smart_195_raw', 'smart_196_normalized',
          'smart_196_raw', 'smart_197_normalized', 'smart_197_raw',
          'smart_198_normalized', 'smart_198_raw', 'smart_199_normalized',
          'smart_199_raw', 'smart_200_normalized', 'smart_200_raw',
          'smart_201_normalized', 'smart_201_raw', 'smart_220_normalized',
          'smart_220_raw', 'smart_222_normalized', 'smart_222_raw',
          'smart_223_normalized', 'smart_223_raw', 'smart_224_normalized',
          'smart_224_raw', 'smart_225_normalized', 'smart_225_raw',
          'smart_226_normalized', 'smart_226_raw', 'smart_240_normalized',
          'smart_240_raw', 'smart_241_normalized', 'smart_241_raw',
          'smart_242_normalized', 'smart_242_raw', 'smart_250_normalized',
          'smart_250_raw', 'smart_251_normalized', 'smart_251_raw',
          'smart_252_normalized', 'smart_252_raw', 'smart_254_normalized',
          'smart_254_raw', 'smart_255_normalized', 'smart_255_raw']
    dset = pd.read_csv(
        '/Users/gpistre/Prevision/prevision-python/examples/data/mclass.csv'
    ).sample(n=101).rename(columns={'failure': 'target'}).drop(cs, axis=1)

    uc_config = pio.TrainingConfig(models=[pio.Model.LinReg],

                                   features=[pio.Feature.Counts],

                                   profile=pio.Profile.Quick,
                                   with_blend=False)

    col_config = pio.ColumnConfig(target_column='target')

    train_dset = pio.Dataset.new(name='events_test' + '_train',
                                 dataframe=dset)

    uc = pio.MultiClassification.fit('events_test',
                                     dataset=train_dset,
                                     column_config=col_config,
                                     training_config=uc_config)

    uc.save()

    uc = pio.Supervised.load('events_test.pio')
    uc.wait_until(lambda u: len(u) > 0, timeout=None)

    preds = uc.predict(dset.drop('target', axis=1))
    print(preds.head())
