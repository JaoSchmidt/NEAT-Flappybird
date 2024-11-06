#pragma once
#include "Core.h"
#include "NEAT/Activation.h"

#include <cmath>
#include <pain.h>
#include <vector>

class Individual;

struct InnovationStatic {
  int m_innovation;
  int m_InNodeId;
  int m_OutNodeId;

  InnovationStatic(int innovation, int inNodeId, int outNodeId)
      : m_innovation(innovation), m_InNodeId(inNodeId), m_OutNodeId(outNodeId)
  {
  }
  InnovationStatic(InnovationStatic &&o) = default;
  InnovationStatic &operator=(InnovationStatic &&o) = default;

  COPIES(InnovationStatic);
};

struct ConnectionGene {
  int m_InNodeId;
  int m_OutNodeId;
  int m_innovation;
  double m_weight;
  bool m_enable;

  ConnectionGene(int inputNodeId, int outputNodeId, double weight, bool enable,
                 int innovation)
      : m_InNodeId{inputNodeId}, m_OutNodeId{outputNodeId},
        m_innovation{innovation}, m_weight{weight}, m_enable{enable}
  {
  }

  ConnectionGene(ConnectionGene &&o) = default;
  ConnectionGene &operator=(ConnectionGene &&o) = default;

  ConnectionGene clone() const
  {
    return ConnectionGene(m_InNodeId, m_OutNodeId, m_weight, m_enable,
                          m_innovation);
  }

  COPIES(ConnectionGene)
};

// represents a neuron
struct NodeGene {
  int m_neuron_id;
  double m_bias;
  ActivationFunction m_activationFunction;
  int m_layer_id;
  NodeGene(int nodeId, double bias, int layerId)
      : m_neuron_id{nodeId}, m_bias{bias}, m_layer_id{layerId}
  {
  }

  NodeGene(NodeGene &&o) = default;
  NodeGene &operator=(NodeGene &&o) = default;
  // copy
  NodeGene clone() const { return NodeGene(m_neuron_id, m_bias, m_layer_id); };
  COPIES(NodeGene)
};

// all genes
struct Genome {
  std::vector<NodeGene> m_neurons;
  std::vector<ConnectionGene> m_links;
  std::vector<int> m_layers;
  std::vector<InnovationStatic> &m_globalInnovations;

  void addConnection(ConnectionGene connGene);
  void addNode(ConnectionGene &oldGene, int numInputs, int numOutputs);
  void removeConnection(ConnectionGene &connGene);
  void removeNode(NodeGene &nodeGene);
  // Non Structural Mutations
  void setBias(int neuron_id, double bias);
  void setWeight(int link_id, double weight);
  std::vector<double> run(const std::vector<double> &inputs, int numInputs,
                          int numOutputs);

  // get correct innovation or create new one
  int loadInnovation(int inputNodeId, int outputNodeId);

private:
  friend class Individual;
  int getLayerIndexDistance(int firstLayerId, int secondLayerId)
  {
    // clang-format off
    auto inLayerIt = std::find(m_layers.begin(), m_layers.end(), firstLayerId);
    ASSERT(inLayerIt != m_layers.end(), "firstLayerID {} not found in m_layers",firstLayerId);
    auto outLayerIt =std::find(m_layers.begin(), m_layers.end(), secondLayerId);
    ASSERT(outLayerIt != m_layers.end(),"secondLayerID {} not found in m_layers", secondLayerId);
    // clang-format on

    return std::distance(inLayerIt, outLayerIt);
  }
  const int getNodeLayer(int nodeId) const
  {
    // Equivalent to m_neurosn.find() in JS
    auto nodeIt = std::find_if(
        m_neurons.begin(), m_neurons.end(),
        [&](const NodeGene &node) { return node.m_neuron_id == nodeId; });

    ASSERT(nodeIt != m_neurons.end(), "Input  ID {} not found in neurons",
           nodeId);
    return nodeIt->m_layer_id;
  }
  int &getNodeLayer(int nodeId)
  {
    // Equivalent to m_neurosn.find() in JS
    auto nodeIt = std::find_if(
        m_neurons.begin(), m_neurons.end(),
        [&](const NodeGene &node) { return node.m_neuron_id == nodeId; });

    ASSERT(nodeIt != m_neurons.end(), "Input  ID {} not found in neurons",
           nodeId);
    return nodeIt->m_layer_id;
  }
  int lowerLayerNodeAdition(int inputNodeId, int outputNodeId);
  int sameLayerLinkAdition(int nodeId);

public:
  ~Genome() = default;
  Genome(std::vector<InnovationStatic> &globalInnovations)
      : m_neurons{}, m_links{}, m_layers{},
        m_globalInnovations{globalInnovations}
  {
  }

  Genome(std::vector<NodeGene> neurons, std::vector<ConnectionGene> links,
         std::vector<int> layers,
         std::vector<InnovationStatic> &globalInnovations)
      : m_neurons(std::move(neurons)), m_links(std::move(links)),
        m_layers(std::move(layers)), m_globalInnovations(globalInnovations)
  {
  }

  // Copy Constructor
  Genome(const Genome &other)
      : m_neurons(other.m_neurons), m_links(other.m_links),
        m_layers(other.m_layers), m_globalInnovations(other.m_globalInnovations)
  {
  }
  Genome &operator=(const Genome &other)
  {
    if (this != &other) {
      m_neurons = other.m_neurons;
      m_links = other.m_links;
      m_layers = other.m_layers;
      m_globalInnovations = other.m_globalInnovations;
    }
    LOG_I("globalInnovations {}", m_globalInnovations.size());
    return *this;
  }
  Genome(Genome &&other) noexcept
      : m_neurons(std::move(other.m_neurons)),
        m_links(std::move(other.m_links)), m_layers(std::move(other.m_layers)),
        m_globalInnovations(other.m_globalInnovations)
  {
  }
  Genome &operator=(Genome &&other) noexcept
  {
    if (this != &other) {
      m_neurons = std::move(other.m_neurons);
      m_links = std::move(other.m_links);
      m_layers = std::move(other.m_layers);
      m_globalInnovations = other.m_globalInnovations;
    }
    return *this;
  }
};
