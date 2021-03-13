
from .corpora import *


CORPUS_LIST = {
    'EMVIC2014': EMVIC2014,
    'Cerf2007-FIFA': Cerf2007_FIFA,
    'ETRA2019': ETRA2019,
    'ETRA2019-Fixation': ETRA2019_Fixation,
    # 'IRCCyN_IVC_Eyetracker_Images_LIVE_Database': IRCCyN_IVC_LIVE,
    # 'Dorr2010-GazeCom': Dorr2010_GazeCom,
    'MIT-LearningToPredict': MIT_LearningToPredict,
    'MIT-LowRes': MIT_LowRes,
    'MIT-CVCL': MIT_CVCL,
    # 'Kubota2012-WideField': Kubota2012_WideField,
    # 'Sugano2013-BSDS500': Sugano2013_BSDS500,
    # 'IRCCyN_IVC_Eyetracker_Berkeley_Database': IRCCyN_IVC_Berkeley,
    # 'DOVES': DOVES()
}


def get_corpora(args, additional_corpus=None):
    if args.task:
        return {corpus: CORPUS_LIST[corpus](args)
                for corpus in TASK_CORPORA[args.task]}

    corpora = list(CORPUS_LIST.keys())

    # used to add a corpus to evaluator to test for overfitting during
    # training time
    if isinstance(additional_corpus, list):
        for c in additional_corpus:
            if c not in corpora:
                corpora.append(c)
                logging.info('[evaluator] Added an unseen data set: {}'.format(c))
    return {c: CORPUS_LIST[c](args) for c in corpora}
