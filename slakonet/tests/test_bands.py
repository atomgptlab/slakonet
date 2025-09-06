from slakonet.get_bands import get_gap
from slakonet.optim import MultiElementSkfParameterOptimizer
import os

model_path=os.path.join(os.path.dirname(__file__),"slakonet_v1_sic")
def test_si():

 model = MultiElementSkfParameterOptimizer.load_model(model_path, method='state_dict')
 get_gap(jid='JVASP-1002',model=model,plot=True)

# test_si()
