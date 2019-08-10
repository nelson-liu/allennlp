from allennlp.common.util import JsonDict
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('tagger')
class TaggerPredictor(Predictor):
    """"
    Predictor wrapper for the Tagger.
    """
    def dump_line(self, outputs: JsonDict) -> str:
        if "mask" in outputs:
            return str(outputs["tags"][:sum(outputs["mask"])]) + "\n"
        else:
            return str(outputs["tags"]) + "\n"
