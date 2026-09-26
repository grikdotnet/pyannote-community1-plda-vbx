# Preserve speech when the speaker limit forces a merge

Local speakers with a temporal conflict should receive different global identities when the configured speaker limit permits it. If no existing identity can be reused without a conflict, create one while capacity remains. Once the maximum is reached, assign the local speaker to the highest-scoring existing identity even if that merges simultaneous speakers. This keeps detected speech in the output and respects the configured limit, while accepting that identity accuracy can degrade when the constraints are incompatible.
