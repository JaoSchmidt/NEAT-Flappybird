#include "Core.h"
#include "NEAT/Activation.h"

#include <cmath>
#include <pain.h>
#include <vector>

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

private:
  COPIES(InnovationStatic);
};

struct ConnectionGene {
  int m_InNodeId;
  int m_OutNodeId;
  double m_weight;
  bool m_enable;
  int m_innovation;

  ConnectionGene(int inputNodeId, int outputNodeId, double weight, bool enable,
                 int innovation)
      : m_InNodeId{inputNodeId}, m_OutNodeId{outputNodeId}, m_weight{weight},
        m_enable{enable}, m_innovation{innovation}
  {
  }

  ConnectionGene(ConnectionGene &&o) = default;
  ConnectionGene &operator=(ConnectionGene &&o) = default;

private:
  COPIES(ConnectionGene); // set copy constructor and assign as private
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

private:
  COPIES(NodeGene); // set copy constructor and assign as private
};

// all genes
struct Genome {
  int m_genome_id;
  static int m_num_inputs;
  static int m_num_ouputs;
  std::vector<NodeGene> m_neurons;
  std::vector<ConnectionGene> m_links;
  std::vector<int> m_layers;
  static std::vector<InnovationStatic> m_globalInnovations;

  void addConnection(ConnectionGene connGene);
  void addNode(ConnectionGene &oldGene);
  void removeConnection(ConnectionGene &connGene);
  void removeNode(NodeGene &nodeGene);
  std::vector<double> run(const std::vector<double> &inputs);

  // clang-format off
  Genome(int genome_id, int num_inputs, int num_outputs)
      : m_genome_id(genome_id){}
  Genome(Genome &&o)
      : m_genome_id(o.m_genome_id), m_neurons(std::move(o.m_neurons)),
        m_links(std::move(o.m_links)){}
  // clang-format on

private:
  int getLayerIndexDistance(int inLayerId, int outLayerId)
  {
    auto inLayerIt = std::find(m_layers.begin(), m_layers.end(), inLayerId);
    ASSERT(inLayerIt != m_layers.end(), "inLayerID {} not found in m_layers",
           inLayerId);
    auto outLayerIt = std::find(m_layers.begin(), m_layers.end(), outLayerId);
    ASSERT(outLayerIt != m_layers.end(), "outLayerID {} not found in m_layers",
           outLayerId);

    return std::distance(inLayerIt, outLayerIt);
  }
  const int getNodeLayer(int nodeId) const
  {
    // Equivalent to m_neurosn.find() in JS
    auto nodeIt = std::find_if(
        m_neurons.begin(), m_neurons.end(),
        [&](const NodeGene &node) { return node.m_neuron_id == nodeId; });

    ASSERT(nodeIt != m_neurons.end(), "Input  ID {} not found in neurons");
    return nodeIt->m_layer_id;
  }
  int &getNodeLayer(int nodeId)
  {
    // Equivalent to m_neurosn.find() in JS
    auto nodeIt = std::find_if(
        m_neurons.begin(), m_neurons.end(),
        [&](const NodeGene &node) { return node.m_neuron_id == nodeId; });

    ASSERT(nodeIt == m_neurons.end(), "Input  ID {} not found in neurons");
    return nodeIt->m_layer_id;
  }
  int needLowerNode(int inputNodeId);
  COPIES(Genome); // set copy constructor and assign as private
};
