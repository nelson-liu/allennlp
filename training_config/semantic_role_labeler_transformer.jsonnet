// Configuration for a semantic role labeler model based on:
//   He, Luheng et al. “Deep Semantic Role Labeling: What Works and What's Next.” ACL (2017).
{
  "dataset_reader":{"type":"srl"},
  "train_data_path": "/data/conll-formatted-ontonotes-5.0/data/train",
  "validation_data_path": "/data/conll-formatted-ontonotes-5.0/data/development",
  "test_data_path": "/data/conll-formatted-ontonotes-5.0/data/test",
  "evaluate_on_test": true,
  "model": {
    "type": "srl",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 100,
            "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
            "trainable": true
        }
      }
    },
    "initializer": [
      [
        "tag_projection_layer.*weight",
        {
          "type": "orthogonal"
        }
      ]
    ],
    "encoder": {
      "type": "stacked_self_attention",
      "input_dim": 200,
      "hidden_dim": 200,
      "projection_dim": 200,
      "num_layers": 10,
      "feedforward_hidden_dim": 800,
      "num_attention_heads": 8
    },
    "binary_feature_dim": 100,
    "label_smoothing" : 0.1
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size" : 80
  },

  "trainer": {
    "num_epochs": 500,
    "grad_clipping": 1.0,
    "patience": 20,
    "validation_metric": "+f1-measure-overall",
    "cuda_device": 0,
    "learning_rate_scheduler": {
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "mode": "max",
        "patience": 2
    },
    "optimizer": {
      "type": "adadelta",
      "rho": 0.95
    }
  }
}
