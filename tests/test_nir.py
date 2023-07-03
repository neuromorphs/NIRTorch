import torch
import torch.nn as nn

def test_extract_nir_edges():
    mymodel = nn.Sequential(
        nn.Conv2d(2, 10, 3),
        nn.ReLU(),
        nn.AvgPool2d(2),
        nn.Conv2d(10, 4, 3),
        nn.ReLU(),
        nn.AvgPool2d(2)
    )    
    sample_data = torch.rand(1, 2, 16, 16)

    from nirtorch.to_nir import extract_nir_graph
    from nir.ir import NIRNode

    def dummy_model_map(module: nn.Module)->NIRNode:
        return NIRNode() 

    nir_graph = extract_nir_graph(mymodel, dummy_model_map, sample_data, "mysequential")
    print(nir_graph)

    assert len(nir_graph.nodes) == 6
    assert len(nir_graph.edges) == 5
    