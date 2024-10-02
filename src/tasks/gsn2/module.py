
from src.common.module import BaseModule
from src.tasks.gsn2.anchor_set import AnchorSet
from src.tasks.gsn2.arch import MyNet32
from src.tasks.gsn2.dataset import ImagesDataset


class ObjectDetectionModule(BaseModule):
    def __init__(self, config=None):
        super(ObjectDetectionModule, self).__init__(config)

        anchor_sizes = [tuple(e) for e in config.model.anchor_set.anchor_sizes]
        anchor_set = AnchorSet(anchor_sizes, k_grid=config.model.anchor_set.k_grid)

        self.model = MyNet32(
            n_layers_backbone=config.model.n_layers_backbone,
            n_layers_clf_head=config.model.n_layers_clf_head,
            n_layers_reg_head=config.model.n_layers_reg_head,
            anchor_sizes=anchor_sizes,
            anchors=anchor_set.list_mnistboxes
        )

        self.ds_train = ImagesDataset("train", config.dataset.train.size)
        self.ds_val = ImagesDataset("val")

        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError
        # x, targets = batch['x'], batch['y_shapes']
        # logits = self.model(x)
        # loss = bcewithlogits_multilabel(logits, targets)
        # return loss

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
        # x, targets = batch['x'], batch['y_shapes']
        # logits = self.model(x)
        # return {"logits": logits, "targets": targets}

    def validation_epoch_end(self, outputs):
        raise NotImplementedError
        # logits = torch.cat([batch['logits'] for batch in outputs])
        # targets = torch.cat([batch['targets'] for batch in outputs])
        #
        # preds = torch.sigmoid(logits)
        # preds_binary = convert_topk_to_binary(preds, 2)
        #
        # acc = (preds_binary.int() == targets).all(dim=1).float().mean()
        # print(f"\n Val/Acc = {acc:.2f}")
        # self.log(f"Val/Acc", acc)

