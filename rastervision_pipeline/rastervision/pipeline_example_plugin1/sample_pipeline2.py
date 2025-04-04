from os.path import join

from rastervision.pipeline.config import Config, register_config
from rastervision.pipeline.file_system import file_to_str, str_to_file
from rastervision.pipeline.pipeline import Pipeline
from rastervision.pipeline.pipeline_config import PipelineConfig
from rastervision.pipeline.utils import split_into_groups


@register_config('pipeline_example_plugin1.message_maker')
class MessageMakerConfig(Config):
    greeting: str = 'hello'

    def build(self):
        return MessageMaker(self)


class MessageMaker:
    def __init__(self, config) -> None:
        self.config = config

    def make_message(self, name) -> str:
        # Use the greeting field to make the message.
        return f'{self.config.greeting} {name}!'


@register_config('pipeline_example_plugin1.sample_pipeline2')
class SamplePipeline2Config(PipelineConfig):
    names: list[str] = ['alice', 'bob']
    message_uris: list[str] | None = None
    # Fields can have other Configs as types.
    message_maker: MessageMakerConfig = MessageMakerConfig()

    def build(self, tmp_dir):
        return SamplePipeline2(self, tmp_dir)

    def update(self) -> None:
        if self.message_uris is None:
            self.message_uris = [
                join(self.root_uri, f'{name}.txt') for name in self.names
            ]


class SamplePipeline2(Pipeline):
    commands: list[str] = ['save_messages', 'print_messages']
    split_commands = ['save_messages']
    gpu_commands = []

    def save_messages(self, split_ind=0, num_splits=1) -> None:
        message_maker = self.config.message_maker.build()

        split_groups = split_into_groups(
            list(zip(self.config.names, self.config.message_uris, strict=False)), num_splits
        )
        split_group = split_groups[split_ind]

        for name, message_uri in split_group:
            # Unlike before, we use the message_maker to make the message.
            message = message_maker.make_message(name)
            str_to_file(message, message_uri)

    def print_messages(self) -> None:
        for message_uri in self.config.message_uris:
            file_to_str(message_uri)
