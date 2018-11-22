// Configuration for an Elmo-augmented coreference resolution model based on:
//   Lee, Kenton et al. “End-to-end Neural Coreference Resolution.” EMNLP (2017).
// Note that this configuration does not reproduce the result in the ELMo paper, see
//   https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md#reproducing-the-results-from-deep-contextualized-word-representations
// for more details.
{
  "dataset_reader": {
    "type": "coref",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": false
      },
      "elmo": {
        "type": "elmo_characters"
      },
      "token_characters": {
        "type": "characters"
      }
    },
    "max_span_width": 10
  },
  "train_data_path": "/data/train.english.v4_gold_conll",
  "validation_data_path": "/data/dev.english.v4_gold_conll",
  "test_data_path": "/data/test.english.v4_gold_conll",
  "evaluate_on_test": true,
  "model": {
    "type": "coref",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.300d.txt.gz",
            "embedding_dim": 300,
            "trainable": false
        },
        "elmo": {
            "type": "elmo_token_embedder",
            "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/4x4096_512_2048cnn_2xhighway/options.json",
            "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/4x4096_512_2048cnn_2xhighway/elmo_4x4096_512_2048cnn_2xhighway_weights.hdf5",
            "do_layer_norm": false,
            "dropout": 0.0
        },
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
            "num_embeddings": 262,
            "embedding_dim": 16
            },
            "encoder": {
            "type": "cnn",
            "embedding_dim": 16,
            "num_filters": 100,
            "ngram_filter_sizes": [5]
            }
        }
      }
    },
    "context_layer": {
        "type": "lstm",
        "bidirectional": true,
        "input_size": 1424,
        "hidden_size": 200,
        "num_layers": 1,
        "dropout": 0.2
    },
    "mention_feedforward": {
        "input_dim": 2244,
        "num_layers": 2,
        "hidden_dims": 150,
        "activations": "relu",
        "dropout": 0.2
    },
    "antecedent_feedforward": {
        "input_dim": 6752,
        "num_layers": 2,
        "hidden_dims": 150,
        "activations": "relu",
        "dropout": 0.2
    },
    "initializer": [
        [".*linear_layers.*weight", {"type": "xavier_normal"}],
        [".*scorer._module.weight", {"type": "xavier_normal"}],
        ["_distance_embedding.weight", {"type": "xavier_normal"}],
        ["_span_width_embedding.weight", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_ih.*", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_hh.*", {"type": "orthogonal"}]
    ],
    "lexical_dropout": 0.5,
    "feature_size": 20,
    "max_span_width": 10,
    "spans_per_word": 0.4,
    // "max_antecedents": 100
    "max_antecedents": 75
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["text", "num_tokens"]],
    "padding_noise": 0.0,
    "batch_size": 1
  },
  "trainer": {
    "num_epochs": 150,
    "grad_norm": 5.0,
    "patience" : 10,
    "cuda_device" : 0,
    "validation_metric": "+coref_f1",
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 2
    },
    "optimizer": {
      "type": "adam"
    }
  }
}
