from allennlp.common.util import JsonDict
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('streusle-tagger')
class StreusleTaggerPredictor(Predictor):
    """"
    Predictor wrapper for the StreusleTagger.
    """
    def dump_line(self, outputs: JsonDict) -> str:
        if "mask" in outputs:
            return str(outputs["tags"][:sum(outputs["mask"])]) + "\n"
        else:
            return str(outputs["tags"]) + "\n"
