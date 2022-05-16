training_args = {
  
        "do_train": True,

        "per_device_train_batch_size": 128,

        "per_device_eval_batch_size": 256,

        "learning_rate": 0.01,

        "weight_decay": 1e-5,

        "num_train_epochs": 10,

        "logging_strategy": "epoch",

        "seed": 420,

        "output_dir": "~/code/",

        "do_eval": True,

        "dataloader_num_workers": 4,

        "evaluation_strategy": "steps",

        "logging_dir": "./logs",

        "logging_strategy": "steps",

        "logging_steps": 1000

}

