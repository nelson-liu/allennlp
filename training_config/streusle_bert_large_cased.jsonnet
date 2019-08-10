{

  "dataset_reader": {
    "type": "streusle",
    "token_indexers": {
        "bert": {
            "type": "bert-pretrained",
            "pretrained_model": "bert-large-cased"
        }
    }
  },
  "train_data_path": "https://raw.githubusercontent.com/nert-nlp/streusle/master/train/streusle.ud_train.json",
  "validation_data_path": "https://raw.githubusercontent.com/nert-nlp/streusle/master/dev/streusle.ud_dev.json",
  "test_data_path": "https://raw.githubusercontent.com/nert-nlp/streusle/master/test/streusle.ud_test.json",
  "model": {
    "type": "streusle_tagger",
    "text_field_embedder": {
        "allow_unmatched_keys": true,
        "embedder_to_indexer_map": {
            "bert": ["bert", "bert-offsets"],
        },
        "token_embedders": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": "bert-large-cased",
                "top_layer_only": true,
                "requires_grad": false
            }
        }
    },
    "regularizer": [
      [
        "scalar_parameters",
        {
          "type": "l2",
          "alpha": 0.1
        }
      ]
    ]
  },
  "iterator": {
    "type": "basic",
    "batch_size": 64
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": 0.001
    },
    "num_serialized_models_to_keep": 3,
    "num_epochs": 75,
    "grad_norm": 5.0,
    "patience": 25,
    "cuda_device": 0
  }
}
