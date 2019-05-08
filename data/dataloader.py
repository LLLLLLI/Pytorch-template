import torch
import torch.utils.data as data

class PreprocessDataset(data.DataLoader):
    def __init__(self, opt, split):
        super(PreprocessDataset, self).__init__()

        self.xx = opt.xx
        self.xxx = opt.xxx
        self.split = split

        self.database = {}

        if self.split == 'train':
            # preprocess the data before using it.
            # save the value_1 of item_1 in database[item_1]['value_1']
            # if it is too big to store in memory, just save the dir and load them in __getitem__()
            pass
        elif self.split == 'val':
            pass
        elif self.split == 'test':
            pass
        else:
            print('Unkwon dataset split')
        
        self.id_list = [id for id in self.database]
        self.length = len(self.id_list)
    
    def __getitem__(self, index):
        name = self.id_list[index]
        feat = self.database[name]['feat']
        label = self.database[name]['label']
        return feat, label
    
    def __len__(self):
        return self.length

def collate_fn(data):
    # Sort a data list by caption length
    images, labels = zip(*data)
    # Merge images (convert tuple of 3D tensor to 4D tensor)
    # Useful if merge data in special way.
    # The following code is the default way
    images = torch.cat(images, 0)
    labels = torch.cat(labels, 0)
    
    return images, labels


def get_dataloader(opt):
    data_train = PreprocessDataset(opt, 'train')
    data_val = PreprocessDataset(opt, 'val')
    train_dataloader = data.DataLoader(dataset=data_train,
                                       batch_size=opt.batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       collate_fn=collate_fn)
    val_dataloader = data.DataLoader(dataset=data_val,
                                      batch_size=1,
                                      shuffle=False,
                                      pin_memory=True,
                                      collate_fn=collate_fn)
    return train_dataloader, val_dataloader
