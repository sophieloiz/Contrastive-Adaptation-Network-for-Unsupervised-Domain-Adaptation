import torch.utils.data
from torch.utils.data import DataLoader
from . import single_dataset
from data import (
    get_transforms,
    load_data_test,
    return_dataset,
)


class CustomDatasetDataLoaderCaps(object):
    def name(self):
        return "CustomDatasetDataLoaderCaps"

    def __init__(
        self,
        dataset_type,
        train,
        batch_size,
        dataset_root="",
        transform=None,
        classnames=None,
        paths=None,
        labels=None,
        num_workers=0,
        **kwargs
    ):

        self.train = train
        self.dataset = getattr(single_dataset, dataset_type)()
        self.dataset.initialize(
            root=dataset_root,
            transform=transform,
            classnames=classnames,
            paths=paths,
            labels=labels,
            **kwargs
        )

        self.classnames = classnames
        self.batch_size = batch_size

        dataset_len = len(self.dataset)
        cur_batch_size = min(dataset_len, batch_size)
        assert cur_batch_size != 0, "Batch size should be nonzero value."

        if self.train:
            drop_last = True
            sampler = torch.utils.data.RandomSampler(self.dataset)
            batch_sampler = torch.utils.data.BatchSampler(
                sampler, self.batch_size, drop_last
            )
        else:
            drop_last = False
            sampler = torch.utils.data.SequentialSampler(self.dataset)
            batch_sampler = torch.utils.data.BatchSampler(
                sampler, self.batch_size, drop_last
            )

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_sampler=batch_sampler, num_workers=int(num_workers)
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    # def prepare_loader(self):
    #     data_train = return_dataset(
    #             self.caps_directory,
    #             split_df_dict["train"],
    #             self.preprocessing_dict,
    #             train_transformations=train_transforms,
    #             all_transformations=all_transforms,
    #             multi_cohort=self.multi_cohort,
    #             label=self.label,
    #             label_code=self.label_code,
    #         )

    #     data_valid = return_dataset(
    #             self.caps_directory,
    #             split_df_dict["validation"],
    #             self.preprocessing_dict,
    #             train_transformations=train_transforms,
    #             all_transformations=all_transforms,
    #             multi_cohort=self.multi_cohort,
    #             label=self.label,
    #             label_code=self.label_code,
    #         )

    #     train_sampler = self.task_manager.generate_sampler(data_train, self.sampler)

    #     train_loader = DataLoader(
    #             data_train,
    #             batch_size=self.batch_size,
    #             sampler=train_sampler,
    #             num_workers=self.n_proc,
    #             worker_init_fn=pl_worker_init_function,
    #         )
    #     valid_loader = DataLoader(
    #             data_valid,
    #             batch_size=self.batch_size,
    #             shuffle=False,
    #             num_workers=self.n_proc,
