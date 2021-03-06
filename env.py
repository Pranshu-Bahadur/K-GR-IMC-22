import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


exec(open('utils.py', 'r').read())
exec(open('run_configs/experiment.bat', 'r').read())


t_dir = '/home/pranshu-bahadur/datasets/train'

training_dataset = TrainIMC22Dataset(t_dir)

splits = split_dataset(training_dataset, 0.8)

training_args = TrainingArguments(**training_args)




torch.multiprocessing.set_sharing_strategy('file_system')



class IMC22Trainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_collator = torch.utils.data.default_collate
        self.loss_fct = torch.nn.MSELoss()

    def compute_loss(self, model, inputs, return_outputs=True):
       """
       Inputs: Batch x Tuple[List[Tensor], Tensor]
       """
       x, y = inputs
       outputs = model(x)
       loss = self.loss_fct(outputs.view(y.size(0),-1), y)
       return loss



class IMC22Model(torch.nn.Module):
    def __init__(self, model_name, **kwargs):
        super().__init__()
        self.model1 = create_model(model_name, num_classes=1, pretrained=True, in_chans=3)
        self.model1.fc = torch.nn.Linear(self.model1.classifier.in_features, 1)
        self.model2 = create_model(model_name, num_classes=1, pretrained=True, in_chans=3)
        self.model2.fc = torch.nn.Linear(self.model2.classifier.in_features, 1)

    def forward(self, x : List[Tensor]):
        print(x.size())
        return self.model1(x[:, 0, :, :, :]) + self.model2(x[:, 1, :, :, :])


train_args = {
        'model': IMC22Model('tf_efficientnetv2_s'),
        'train_dataset': splits[0],
        'eval_dataset': splits[1],
        'args': training_args
}



IMC22Trainer(**train_args).train()
