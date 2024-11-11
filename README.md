# HyperDimRed
A Python library for hyperbolic dimensionality reduction

## Parameters help to run the code

### scatterplot_2d function
*color_by* parameter:
- entropy: is the entropy measuring the uncertainty of the smelling perception, makes sense for continuous values (keller and sagar datasets)
- cid: is the ID of the molecule (keller and sagar datasets)
- distance: is the Euclidean norm of the embedding
- color: is a pre-constructed color for entropy based on intensity, relevant for mixture of odors (ravia dataset)

*shape_by* parameter:
- subject: gives different shapes to the dots depending on the subject
- descriptor: gives different shapes to the dots corresponding to the maximal value of its descriptors (i.e. the most graded perception odor)
