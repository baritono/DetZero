import subprocess
from pathlib import Path

from .version import __version__
from .structures import (
    AugMatrixInv,
    MultiScale3DFeatures,
    MultiScale3DStrides,
    PointFeaturesDict,
    PointCoordsDict,
    DataDict,
    BatchDict,
    SeparateHeadPredDict,
    PredictionDict,
    CenterHeadTargetDict,
    ProposalTargetDict,
    RoIHeadForwardDict,
    ModelInfoDict,
    AnnotationDict,
    RecallDict,
)

__all__ = [
    '__version__',
    'AugMatrixInv',
    'MultiScale3DFeatures',
    'MultiScale3DStrides',
    'PointFeaturesDict',
    'PointCoordsDict',
    'DataDict',
    'BatchDict',
    'SeparateHeadPredDict',
    'PredictionDict',
    'CenterHeadTargetDict',
    'ProposalTargetDict',
    'RoIHeadForwardDict',
    'ModelInfoDict',
    'AnnotationDict',
    'RecallDict',
]


def get_git_commit_number():
    if not (Path(__file__).parent / '../../.git').exists():
        return '0000000'

    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number


script_version = get_git_commit_number()


if script_version not in __version__:
    __version__ = __version__ + '+py%s' % script_version
