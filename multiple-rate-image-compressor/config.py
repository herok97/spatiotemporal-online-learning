class Config():
    def __init__(self):
        # Directory
        self.root_dir = r'C:\Users\duddn\PycharmProjects\compressai/'
        self.save_dir = self.root_dir + 'spatiotemporal/multiple-rate-image-compressor/save/'
        self.log_dir = self. root_dir + 'spatiotemporal/multiple-rate-image-compressor/log/'
        self.dataset = self.root_dir + 'data/'
        # Pretrain ro test
        self.pre_train = True
        self.save_model_dir = self.root_dir + 'spatiotemporal/multiple-rate-image-compressor/save/1630336216_300000.pkl'    # model name
        self.test_batch_size = 1

        # training condition
        self.lr_decay = 0.1
        self.decay_step = 500000
        self.batch_size = 2
        self.epochs = 50
        self.total_global_step = 700000
        self.lr = 0.0001

        # details
        self.lmbda = [32, 128, 512, 2048, 8192]
        self.num_workers = 0
        self.patch_size = (256, 256)
        self.seed = 123
        self.clip_max_norm = 5.0

        print(self.__dict__)