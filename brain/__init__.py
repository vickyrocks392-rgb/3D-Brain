from brain.base import BrainModel, AdamWOptimizer
from brain.rnn import BaseRNN
from brain.gru import BaseGRU, BaseGRUMemBias
from brain.liquid import CandLiquid
from brain.discrete_leaky_rnn import DiscreteLeakyRNN
from brain.plasticity import (
    PlasticRNN,
    PlasticityRule,
    FrozenPlasticity,
    HebbianPlasticity,
    OjaPlasticity,
    ModulatedPlasticity,
    HomeostaticPlasticity,
)

from brain.predictive_plasticity import (
    PredictiveLearningRule,
    FrozenPredictive,
    LocalPredictivePlasticity,
    NeuromodulatedPredictivePlasticity,
    GradientPredictiveBaseline,
)
from brain.structural_plasticity import StructuralPlasticity, StructuralEvent
from brain.predictive_spatial import PredictiveSpatialRNN
from brain.latent_memory import PersistentLatentModule
from brain.latent_spatial import PersistentLatentSpatialRNN
from brain.representation_analysis import (
    LinearDecodingProbe,
    RepresentationalSimilarityAnalysis,
    StateDynamicsAnalyzer,
)

__all__ = [
    "BrainModel",
    "AdamWOptimizer",
    "BaseRNN",
    "BaseGRU",
    "BaseGRUMemBias",
    "CandLiquid",
    "DiscreteLeakyRNN",
    "PlasticRNN",
    "PlasticityRule",
    "FrozenPlasticity",
    "HebbianPlasticity",
    "OjaPlasticity",
    "ModulatedPlasticity",
    "HomeostaticPlasticity",
    "PredictiveSpatialRNN",
    "PredictiveLearningRule",
    "FrozenPredictive",
    "LocalPredictivePlasticity",
    "NeuromodulatedPredictivePlasticity",
    "GradientPredictiveBaseline",
    "StructuralPlasticity",
    "StructuralEvent",
    "PersistentLatentModule",
    "PersistentLatentSpatialRNN",
    "LinearDecodingProbe",
    "RepresentationalSimilarityAnalysis",
    "StateDynamicsAnalyzer",
    "SpikingNeuronModel",
    "SpikeEvent",
    "LIFNeuronPopulation",
    "IzhikevichNeuronPopulation",
    "SpikePropagationQueue",
    "PlasticityMode",
    "STDPEngine",
    "STDPUpdateRecord",
    "SpikingSpatialNetwork",
    "SpikingLatentSpatialNetwork",
    "NeuronType",
    "EILIFPopulation",
    "EIIzhikevichPopulation",
    "EISpikePropagationQueue",
    "HomeostaticSynapticScaling",
    "EIBalancedSpatialNetwork",
    "EIBalancedLatentSpatialNetwork",
    "EligibilityTraceMode",
    "NeuromodulatorMode",
    "ThreeFactorPlasticityEngine",
    "ThreeFactorSpatialNetwork",
    "TwoCompartmentPopulation",
    "FeedbackMode",
    "DendriticEligibilityMode",
    "TargetedSpatialFeedbackEngine",
    "TwoCompartmentSpatialNetwork",
]

from brain.spiking import (
    SpikingNeuronModel,
    SpikeEvent,
    LIFNeuronPopulation,
    IzhikevichNeuronPopulation,
    SpikePropagationQueue,
)
from brain.stdp import (
    PlasticityMode,
    STDPEngine,
    STDPUpdateRecord,
)
from brain.spiking_spatial import (
    SpikingSpatialNetwork,
    SpikingLatentSpatialNetwork,
)
from brain.ei_dynamics import (
    NeuronType,
    EILIFPopulation,
    EIIzhikevichPopulation,
    EISpikePropagationQueue,
)
from brain.homeostasis import HomeostaticSynapticScaling
from brain.ei_spatial import (
    EIBalancedSpatialNetwork,
    EIBalancedLatentSpatialNetwork,
)
from brain.three_factor import (
    EligibilityTraceMode,
    NeuromodulatorMode,
    ThreeFactorPlasticityEngine,
)
from brain.three_factor_spatial import ThreeFactorSpatialNetwork
from brain.two_compartment import TwoCompartmentPopulation
from brain.targeted_feedback import (
    FeedbackMode,
    DendriticEligibilityMode,
    TargetedSpatialFeedbackEngine,
)
from brain.two_compartment_spatial import TwoCompartmentSpatialNetwork
from brain.predictive_gating import GateMode, PredictiveDendriticGating
from brain.attractor_network import AttractorCouplingMode, RecurrentAttractorCircuit
from brain.action_routing import ActionSelectionCircuit
from brain.phase11_network import Phase11SpatialNetwork

from brain.context_action_interface import (
    ContextActionInterface,
    RepresentationSource,
    ReadoutArchitecture,
    ActionCompetitionMode,
)
from brain.phase12_network import Phase12Network
from brain.spike_encoding import (
    BiophysicalSpikeEncoder,
    SpikeEncodingStrategy,
    HomeostasisMode,
)
from brain.action_persistence import (
    ActionPersistenceController,
    ActionPersistenceState,
)
from brain.biophysical_audit import (
    BiophysicalPathwayAuditor,
    StageMetrics,
)
from brain.phase13_network import Phase13Network
from brain.burst_dynamics import BurstDynamicsEngine, BurstStrategy, BurstRecord
from brain.dendritic_coincidence import DendriticCoincidenceDetector, CoincidenceTriggerType
from brain.temporal_coding import TemporalSpikeDecoder, TemporalCodingScheme
from brain.population_resonance import (
    PopulationResonanceEngine,
    ResonanceRegime,
    SpatialResonanceMode,
    PopulationCodingMode,
)
from brain.information_bottleneck import InformationBottleneckAuditor, StageAuditRecord
from brain.phase14_network import Phase14Network

__all__.extend([
    "TwoCompartmentPopulation",
    "FeedbackMode",
    "DendriticEligibilityMode",
    "TargetedSpatialFeedbackEngine",
    "TwoCompartmentSpatialNetwork",
    "GateMode",
    "PredictiveDendriticGating",
    "AttractorCouplingMode",
    "RecurrentAttractorCircuit",
    "ActionSelectionCircuit",
    "Phase11SpatialNetwork",
    "ContextActionInterface",
    "RepresentationSource",
    "ReadoutArchitecture",
    "ActionCompetitionMode",
    "Phase12Network",
    "BiophysicalSpikeEncoder",
    "SpikeEncodingStrategy",
    "HomeostasisMode",
    "ActionPersistenceController",
    "ActionPersistenceState",
    "BiophysicalPathwayAuditor",
    "StageMetrics",
    "Phase13Network",
    "BurstDynamicsEngine",
    "BurstStrategy",
    "BurstRecord",
    "DendriticCoincidenceDetector",
    "CoincidenceTriggerType",
    "TemporalSpikeDecoder",
    "TemporalCodingScheme",
    "PopulationResonanceEngine",
    "ResonanceRegime",
    "SpatialResonanceMode",
    "PopulationCodingMode",
    "InformationBottleneckAuditor",
    "StageAuditRecord",
    "Phase14Network",
])




