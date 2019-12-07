class Path:
    @staticmethod
    def db_root_dir(dataset):
        if dataset == "pascal":
            return "data/VOC2012"
        elif dataset == "sbd":
            return "data/VOC2012/benchmark_RELEASE"
        elif dataset == "context":
            return "data/context/"
        else:
            print(f"Dataset {dataset} not available.")
            raise NotImplementedError
