from slakonet.get_bands import get_gap
from slakonet.optim import MultiElementSkfParameterOptimizer

def test_si():

 model_path="slakonet_v1_sic"
 model = MultiElementSkfParameterOptimizer.load_model(model_path, method='state_dict')
 get_gap(jid='JVASP-1002',model=model,plot=True)

# test_si()
