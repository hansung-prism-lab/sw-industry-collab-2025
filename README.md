# sw-industry-collab-2025
SW Industry-Academia Collaboration Project (Spring 2025)


flowchart LR
    %% Input
    A[User-Written Text] --> B[Embedding Extractor Module]

    %% Embedding Extractor 내부
    subgraph B[Embedding Extractor Module]
        B1[Position Embedding]
        B2[Segment Embedding]
        B3[Token Embedding]
        B1 --> B4[Self-Attention]
        B2 --> B4
        B3 --> B4
        B4 --> B5[Add & Norm]
        B5 --> B6[Feed-Forward]
        B6 --> B7[Add & Norm]
    end

    %% Profile Prediction Module
    B7 --> C[Profile Prediction Module]

    subgraph C[Profile Prediction Module]
        C1[BERT Embeddings]
        C2[Dense Layer]
        C3[MLP]
        C1 --> C2 --> C3
    end

    %% Output
    C3 --> D[Output\n(ŷ1 ... ŷn)]
