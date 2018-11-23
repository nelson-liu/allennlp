{
    "dataset_reader": {
        "type": "srl",
        "token_indexers": {
            "elmo": {
                "type": "elmo_characters"
            }
        }
    },
    "train_data_path": "/data/train/",
    "validation_data_path": "/data/development/",
    "test_data_path": "/data/conll-2012-test/",
    "evaluate_on_test": true,
    "model": {
        "type": "srl",
        "text_field_embedder": {
            "token_embedders": {
                "elmo": {
                    "type": "transformer_token_embedder",
                    "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/calypso/transformer_6_512_base/options.json",
                    "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/calypso/transformer_6_512_base/model_state_epoch_1.2018-04-13-22-28-11.th",
                    "dropout": 0.0,
                    "scalar_mix_parameters": [0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0]
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
        // NOTE: This configuration is correct, but slow.
        // If you are interested in training the SRL model
        // from scratch, you should use the 'alternating_lstm_cuda'
        // encoder instead.
        "encoder": {
            "type": "alternating_lstm",
            "input_size": 1124,
            "hidden_size": 300,
            "num_layers": 8,
            "recurrent_dropout_probability": 0.1,
            "use_input_projection_bias": false
        },
        "binary_feature_dim": 100,
        "regularizer": [
            [
                ".*scalar_parameters.*",
                {
                    "type": "l2",
                    "alpha": 0.001
                }
            ]
        ]
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [
            [
                "tokens",
                "num_tokens"
            ]
        ],
        "batch_size": 80
    },
    "trainer": {
        "num_epochs": 500,
        "grad_clipping": 1.0,
        "patience": 200,
        "num_serialized_models_to_keep": 10,
        "validation_metric": "+f1-measure-overall",
        "cuda_device": 0,
        "optimizer": {
            "type": "adadelta",
            "rho": 0.95
        }
    }
}
