#include "NEAT/Individuals.h"
#include "pain.h"
#include <cstddef>

// Calculates if should press space or not
bool Individual::fit(const std::vector<double> &inputs)
{

  // std::ostringstream oss;
  // oss << std::fixed << std::setprecision(8);
  // for (auto &input : inputs) {
  //   oss << input << ", ";
  // }
  // LOG_T("[{}]", oss.str());
  std::vector<double> outputs{
      m_genome.run(inputs, m_config.m_numInputs, m_config.m_numOutputs)};
  const bool jump = outputs[0] >= 0.5 ? true : false;

  return jump;
}

// clang-format off
Individual Individual::crossover(
  const Individual &other,
  std::vector<InnovationStatic> &populationInnovs) const
{
  // clang-format on
  // Determine the fitter parent
  const Individual &fitterParent =
      (m_fitness >= other.m_fitness) ? *this : other;
  const Individual &lessFitParent =
      (m_fitness < other.m_fitness) ? *this : other;

  // Create a new genome for the offspring
  Genome offspringGenome{populationInnovs};

  // Crossover nodes with matching neuron IDs
  for (const auto &node : fitterParent.m_genome.m_neurons) {
    auto matchIt = std::find_if(
        lessFitParent.m_genome.m_neurons.begin(),
        lessFitParent.m_genome.m_neurons.end(), [&](const NodeGene &otherNode) {
          return otherNode.m_neuron_id == node.m_neuron_id;
        });

    if (matchIt != lessFitParent.m_genome.m_neurons.end()) {
      // Node with matching ID found, choose biases randomly from parents
      double chosenBias =
          (m_rng.uniform<double>() < 0.5) ? node.m_bias : matchIt->m_bias;
      ActivationFunction chosenActivationFunction =
          (m_rng.uniform<double>() < 0.5) ? node.m_activationFunction
                                          : matchIt->m_activationFunction;
      offspringGenome.m_neurons.emplace_back(node.m_neuron_id, chosenBias);
    } else {
      // No matching node, inherit the node from the fitter parent
      offspringGenome.m_neurons.push_back(node.clone());
    }
  }

  // Crossover links with matching innovations
  for (const auto &link : fitterParent.m_genome.m_links) {
    auto matchIt =
        std::find_if(lessFitParent.m_genome.m_links.begin(),
                     lessFitParent.m_genome.m_links.end(),
                     [&](const ConnectionGene &otherLink) {
                       return otherLink.m_innovation == link.m_innovation;
                     });

    if (matchIt != lessFitParent.m_genome.m_links.end()) {
      // Matching link found, randomly choose weight from either parent
      const double chosenWeight =
          (m_rng.uniform<double>() < 0.5) ? link.m_weight : matchIt->m_weight;
      const bool isEnabled =
          (m_rng.uniform<double>() < 0.5) ? link.m_enable : matchIt->m_enable;
      offspringGenome.m_links.emplace_back(link.m_InNodeId, link.m_OutNodeId,
                                           chosenWeight, isEnabled,
                                           link.m_innovation);
    } else {
      // No matching link, inherit from the fitter parent
      offspringGenome.m_links.push_back(link.clone());
    }
  }

  // Add excess genes from the fitter parent
  for (const auto &excessNode : fitterParent.m_genome.m_neurons) {
    auto matchIt = std::find_if(
        offspringGenome.m_neurons.begin(), offspringGenome.m_neurons.end(),
        [&](const NodeGene &node) {
          return node.m_neuron_id == excessNode.m_neuron_id;
        });

    if (matchIt == offspringGenome.m_neurons.end()) {
      offspringGenome.m_neurons.push_back(excessNode.clone());
    }
  }

  for (const auto &excessLink : fitterParent.m_genome.m_links) {
    auto matchIt = std::find_if(
        offspringGenome.m_links.begin(), offspringGenome.m_links.end(),
        [&](const ConnectionGene &link) {
          return link.m_innovation == excessLink.m_innovation;
        });

    if (matchIt == offspringGenome.m_links.end()) {
      offspringGenome.m_links.push_back(excessLink.clone());
    }
  }

  // Create the offspring Individual with a new genome and reset fitness
  Individual offspring =
      Individual(std::move(offspringGenome), m_config, m_rng);
  offspring.topologySort();
  return offspring;
}

double Individual::calculateDelta(const Individual &other) const
{
  // Coefficients from the configuration
  double c1 = m_config.m_c1; // Assuming this is where c1 is defined
  double c2 = m_config.m_c2; // Example value for c2, adjust as necessary
  double c3 = m_config.m_c3; // Example value for c3, adjust as necessary

  int excessCount = 0;
  int disjointCount = 0;
  double totalWeightDifference = 0.0;
  int matchingCount = 0;

  // Count matching, excess, and disjoint genes
  for (const auto &link : m_genome.m_links) {
    auto otherLinkIt = std::find_if(
        other.m_genome.m_links.begin(), other.m_genome.m_links.end(),
        [&link](const ConnectionGene &otherLink) {
          return otherLink.m_innovation == link.m_innovation;
        });
    if (otherLinkIt != other.m_genome.m_links.end()) {
      // Matching gene
      double weightDifference = std::abs(link.m_weight - otherLinkIt->m_weight);
      totalWeightDifference += weightDifference; // Sum the differences
      matchingCount++;
    }
  }

  // Count excess and disjoint genes
  excessCount = (m_genome.m_links.size() - matchingCount);
  disjointCount = (other.m_genome.m_links.size() - matchingCount);

  // Calculate W, the average weight difference of matching genes
  double W =
      (matchingCount > 0) ? (totalWeightDifference / matchingCount) : 0.0;

  double N = 1.0;
  if (m_genome.m_neurons.size() > 30)
    N = m_genome.m_neurons.size() > other.m_genome.m_neurons.size()
            ? m_genome.m_neurons.size()
            : other.m_genome.m_neurons.size();
  // Calculate delta using the formula
  double delta = (excessCount * c1) / N + (disjointCount * c2) / N + (W * c3);

  return delta;
}

void Individual::mutateAddNeuron()
{
  if (m_rng.uniform<double>(0.0, 1.0) > m_config.m_probAddNode) {
    return; // Mutation does not occur
  }
  // Ensure there are links to choose from
  if (m_genome.m_links.empty()) {
    return; // No connections to mutate
  }

  std::size_t index =
      m_rng.uniform<std::size_t>(0, m_genome.m_links.size() - 1);
  while (m_genome.m_links[index].m_enable == false) {
    index = m_rng.uniform<std::size_t>(0, m_genome.m_links.size() - 1);
  }
  ConnectionGene &selectedConnection = m_genome.m_links[index];

  // Call addNode with the selected connection gene
  m_genome.addNode(selectedConnection, m_config.m_numInputs,
                   m_config.m_numOutputs);
}

void Individual::mutateAddLink()
{
  if (m_rng.uniform<double>(0.0, 1.0) > m_config.m_probAddConn) {
    return; // Mutation does not occur
  }
  // alias for input and output size
  int inTotal = m_config.m_numInputs;

  // Ensure there are neurons to choose from
  if (m_genome.m_neurons.size() < 2) {
    return; // Not enough neurons to connect
  }

  // Select two random neurons
  std::size_t fromIndex =
      m_rng.uniform<std::size_t>(0, m_genome.m_neurons.size() - 1);
  std::size_t toIndex =
      m_rng.uniform<std::size_t>(0, m_genome.m_neurons.size() - 1);

  // Create a new connection
  const int inputId = m_genome.m_neurons[fromIndex].m_neuron_id;
  int outputId = m_genome.m_neurons[toIndex].m_neuron_id;

  // Ensure we don't connect a neuron to itself
  while (inputId == outputId) {
    toIndex = m_rng.uniform<std::size_t>(0, m_genome.m_neurons.size() - 1);
    outputId = m_genome.m_neurons[toIndex].m_neuron_id;
  }

  // check if new link isn't connecting inputs nor connecting outputs
  // HACK: This is hardcoded for the problem. Fix later
  if ((inputId < 0 && outputId < 0) || (inputId > 0 && outputId < 0))
    return;

  // check if there is already a disabled link
  auto linkIt = std::find_if(m_genome.m_links.begin(), m_genome.m_links.end(),
                             [&](const ConnectionGene &link) {
                               return link.m_InNodeId == (int)fromIndex &&
                                      link.m_OutNodeId == (int)toIndex;
                             });
  // if true, just enable it back
  if (linkIt != m_genome.m_links.end()) {
    linkIt->m_enable = true;
  } else {
    ConnectionGene newConnection(
        inputId,        // input neuron id
        outputId,       // output neuron id
        replaceValue(), // weight can be initialized to 0.0 or some random value
        true,           // enabled
        m_genome.loadInnovation(inputId, outputId));

    // Add the new connection to the genome
    m_genome.addConnectionAndSort(std::move(newConnection),
                                  m_config.m_numInputs);
  }
}

void Individual::mutateRemoveNeuron()
{
  if (m_rng.uniform<double>(0.0, 1.0) > m_config.m_probRmNode) {
    return; // Mutation does not occur
  }
  // Ensure there are neurons to choose from
  if (m_genome.m_neurons.size() <=
      static_cast<unsigned>(m_config.m_numInputs)) {
    return; // Can't remove input neurons
  }

  // Select a random neuron that is not an input neuron
  std::size_t index =
      m_rng.uniform<std::size_t>(0, m_genome.m_neurons.size() - 1);

  // Remove the neuron from the genome
  NodeGene &selectedNeuron = m_genome.m_neurons[index];
  m_genome.removeNode(selectedNeuron, m_config.m_numInputs);
}

void Individual::mutateRemoveLink()
{
  if (m_rng.uniform<double>(0.0, 1.0) > m_config.m_probRmConn) {
    return; // Mutation does not occur
  }
  // Ensure there are links to choose from
  if (m_genome.m_links.empty()) {
    return; // No connections to remove
  }

  // Select a random connection gene
  std::size_t index =
      m_rng.uniform<std::size_t>(0, m_genome.m_links.size() - 1);

  // Remove the connection from the genome
  ConnectionGene &selectedLink = m_genome.m_links[index];
  m_genome.removeConnection(selectedLink, m_config.m_numInputs);
}

void Individual::nonStructuralMutate()
{
  // Mutate connection weights
  for (auto &link : m_genome.m_links) {
    if (m_rng.uniform<double>(0.0, 1.0) < m_config.m_mutationRate) {
      // Apply mutation
      if (m_rng.uniform<double>(0.0, 1.0) < m_config.m_replacementRate)
        // Replace with a new value within the allowed range
        link.m_weight = replaceValue();
      else
        // Adjust by a small delta
        link.m_weight = mutateDelta(link.m_weight);
    }
  }

  // Mutate neuron biases
  for (auto &neuron : m_genome.m_neurons) {
    if (m_rng.uniform<double>(0.0, 1.0) < m_config.m_biasMutationRate) {
      // Apply mutation: either replace the bias or adjust it by a delta
      if (m_rng.uniform<double>(0.0, 1.0) < m_config.m_biasReplacementRate)
        // Replace with a new value within the allowed range
        neuron.m_bias = replaceValue();
      else
        // Adjust by a small delta
        neuron.m_bias = mutateDelta(neuron.m_bias);
    }
  }
}
