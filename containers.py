import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from trees.tree import Trees
from cluster import cluster_strategy
from encoder import cent, io_encoder, test_gen
from utils import dataloader, codedataloader, executor, tracer
from utils.tree import GraphRenderer
from utils.other import ConfigManager
from dependency_injector import containers, providers


def select_encoder(trees, tracer, config_manager: ConfigManager, switch=None,  *args, **kwargs):
    encoder_map = {
            "cent": cent.CentEncoder,
            "io": io_encoder.IOEncoder,
        }
    if not switch:
        return encoder_map.get("cent", io_encoder.IOEncoder)(
            trees=trees, tracer=tracer, config_manager=config_manager
        )
    else:
        return encoder_map.get(switch, cent.CentEncoder)(
            trees=trees, tracer=tracer, config_manager=config_manager, max_length=256, 
            static_max_len=kwargs["static_max_len"] if kwargs.get("static_max_len") else 128
        )


def select_cluster(config_manager: ConfigManager, **kwargs):
    return cluster_strategy.Pairwise(config_manager, **kwargs)


class Application(containers.DeclarativeContainer):
    from utils.other import print_split_line
    print_split_line('container init')

    config = providers.Configuration(
        yaml_files=[str(PROJECT_ROOT / "config.yml")]
    )

    config_manager = providers.Singleton(
        ConfigManager,
        config=config
    )

    test_cases = providers.Singleton(
        dataloader.TestCasesLoader.get_test_cases,
        test_cases_folder_path=config.test_case_folder
    )

    dataloader = providers.Singleton(
        codedataloader.CodeDataLoader,
        dataset_path=config.true_code_path,
        code_src=config.code_src,
        batch_size=config.batch_size,
        split_ratio=config.split_ratio,
        dynamic_max_len=128
    )

    executor = providers.Singleton(
        executor.Executor,
        test_cases=test_cases,
        config_manager=config_manager
    )

    trees = providers.Singleton(
        Trees,
        executor=executor,
        config_manager=config_manager
    )

    test_trees = providers.Singleton(
        Trees,
        executor=executor,
        config_manager=config_manager
    )
    

    tracer = providers.Singleton(
        tracer.Tracer,
        executor=executor,
        config_manager=config_manager,
    )

    encoder_io = providers.Singleton(
        io_encoder.IOEncoder,
        trees=trees,
        executor=executor,
        tracer=tracer,
        config_manager=config_manager
    )

    test_fuzzer = providers.Singleton(
        test_gen.TestFuzzer,
        config_manager=config_manager
    )

    encoder_cent = providers.Singleton(
        select_encoder,
        trees=trees,
        tracer=tracer,
        config_manager=config_manager,
        switch="cent"
    )

    strategy = providers.Singleton(
        select_cluster,
        config_manager=config_manager,
        io=encoder_io,
        cent=encoder_cent,
    ) 

    strategy_gp = providers.Singleton(
        cluster_strategy.Pairwise_gp,
        config_manager=config_manager,
        io=encoder_io,
        cent=encoder_cent,
    )

    graphRenderer = providers.Singleton(
        GraphRenderer
    )

    print_split_line('dependencies init done!')
