from torchtitan.config_manager import JobConfig as BaseJobConfig


class JobConfig(BaseJobConfig):

    def __init__(self):
        super().__init__()

        self.parser.add_argument(
            "--training.expert_parallel_degree",
            type=int,
            default=1,
            help="Number of experts to parallelize the model, 1 means disabled",
        )
