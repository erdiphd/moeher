from .normal import NormalLearner
from .hgg import HGGLearner
from .ea import EALearner
learner_collection = {
	'normal': NormalLearner,
	'hgg': HGGLearner,
 	'ea': EALearner,
}

def create_learner(args):
	return learner_collection[args.learn](args)