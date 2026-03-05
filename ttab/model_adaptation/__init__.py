# -*- coding: utf-8 -*-
from .bn_adapt import BNAdapt
from .conjugate_pl import ConjugatePL
from .cotta import CoTTA
from .eata import EATA
from .memo import MEMO
from .no_adaptation import NoAdaptation
from .note import NOTE
from .sar import SAR
from .shot import SHOT
from .t3a import T3A
from .tent import TENT
from .ttt import TTT
from .ttt_plus_plus import TTTPlusPlus
from .rotta import Rotta
from .vida import ViDA
from .deyo import DEYO
from .come import COME
from .adadem import AdaDEM
from .eata_c import EATAC
from .sar2 import SAR2
from .nctta import NCTTA


def get_model_adaptation_method(adaptation_name):
    return {
        "no_adaptation": NoAdaptation,
        "tent": TENT,
        "bn_adapt": BNAdapt,
        "memo": MEMO,
        "shot": SHOT,
        "t3a": T3A,
        "ttt": TTT,
        "ttt_plus_plus": TTTPlusPlus,
        "note": NOTE,
        "sar": SAR,
        "conjugate_pl": ConjugatePL,
        "cotta": CoTTA,
        "eata": EATA,
        "rotta": Rotta,
        "vida": ViDA,
        "deyo": DEYO,
        "come": COME,
        "adadem": AdaDEM,
        "eatac": EATAC,
        "sar2": SAR2,
        "nctta": NCTTA,
    }[adaptation_name]
