"""retina_oct dataset."""

import tensorflow_datasets as tfds
import re

class_label = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
train_url = 'https://drive.google.com/file/d/1AGqDxwCysAvj6CA7oX8hX3EEQCbiRZp6/view?usp=sharing'
test_url = 'https://drive.google.com/file/d/1AGqDxwCysAvj6CA7oX8hX3EEQCbiRZp6/view?usp=sharing'

# TODO(retina_oct): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Retinal OCT (Optical Coherence Tomography) images of Normal, CNV (Choroidal NeoVascularization), DME (Diabetic Macular Edema ) and Drusen condition  
"""

# TODO(retina_oct): BibTeX citation
_CITATION = """
title = {Retinal OCT images}
author = {Paul Mooney}
publisher = {http://Kaggle.com}
url = {https://www.kaggle.com/paultimothymooney/kermany2018}
"""


class RetinaOct(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for retina_oct dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(retina_oct): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(None, None, 3)),
            'label': tfds.features.ClassLabel(names=class_label),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://www.kaggle.com/paultimothymooney/kermany2018',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
     # TODO(retina_oct): Downloads the data and defines the splits
    train_path, test_path = dl_manager.download_and_extract([train_url, test_url])

    # TODO(retina_oct): Returns the Dict[split names, Iterator[Key, Example]]
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            get_kwargs={
                'archive': dl_manager.iter_archive(train_path)
            }
        ), tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            get_kwargs={
                'archive': dl_manager.iter_archive(test_path)
            }
        )
    ]

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(retina_oct): Yields (key, example) tuples from the dataset
    '''
    for f in path.glob('*.jpeg'):
      yield 'key', {
          'image': f,
          'label': 'yes',
      }
    '''
    _name = re.complie((r"^([\w]*[\\/])(CNV|DME|DRUSEN|NORMAL)(?:/|\\)[\w-]*\.jpeg$"))
    for name, obj in path:
        data = _name.match(name)
        if not data:
            continue
        label = data.group(2)
        datarec = {
            "image": obj,
            "label": label
        }
    yield name, record
