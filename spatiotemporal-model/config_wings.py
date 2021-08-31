class Config():
    def __init__(self):
        # Directory
        self.root_dir = r'C:\Users\duddn\PycharmProjects\compressai/'
        self.train_dataset = r'C:\Users\duddn\PycharmProjects\compressai\data\data\vimeo_septuplet\sequences/'
        self.test_dataset = r'C:\Users\duddn\PycharmProjects\compressai\data\data\UVG\images/'
        self.save_dir = self.root_dir + 'spatiotemporal/spatiotemporal-model/save/'
        self.log_dir = self.root_dir + 'spatiotemporal/spatiotemporal-model/log/'
        self.imgCompDir = self.root_dir + 'spatiotemporal/multiple-rate-image-compressor/save/final/1630367628_400000.pkl'  # model name

        # Pretrain ro test
        self.pre_train = True
        self.save_model_dir = self.root_dir + 'spatiotemporal/spatiotemporal-model/save/1630405419_20000.pkl'  # model name
        self.test_batch_size = 1

        # training condition
        self.lr_decay = 0.1
        self.decay_step = 200000
        self.batch_size = 2
        self.epochs = 50
        self.total_global_step = 300000
        self.lr = 0.00005

        # details
        self.lmbda = [32, 128, 512, 2048, 8192]
        self.num_workers = 2
        self.patch_size = (256, 256)
        self.seed = 123
        self.clip_max_norm = 5.0

        print(self.__dict__)
