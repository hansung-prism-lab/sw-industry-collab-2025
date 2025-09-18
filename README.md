# sw-industry-collab-2025
SW Industry-Academia Collaboration Project (Spring 2025)


```mermaid
flowchart TD

    %% 1단계: 데이터 변환
    A[JSON Converter] --> B[final_panel_data.json<br/>36,112 records]

    %% 2단계: AI Processing
    B --> C[Text Generator<br/>Claude 3.5 Haiku]
    B --> D[Embedding Creator<br/>KURE-v1]
    B --> E[Hashtag Generator<br/>Claude 3.5 Haiku]

    C --> F[PostgreSQL 14<br/>with pgvector]
    D --> F
    E --> F

    %% 3단계: Web Service
    F --> G[Phase 3: Web Service v5.0<br/>Rails 8.0.2 백엔드]

    %% 하위 모듈
    G --> H[쿼리 증강<br/>Claude Opus 4.1]
    G --> I[임베딩 서비스<br/>Python FastAPI]
    G --> J[Redis 캐시]
    G --> K[JSONB 필터 엔진]
    G --> L[회로 차단기]
    G --> M[웹 인터페이스<br/>Tailwind + Stimulus]
    G --> N[REST API v1]
    G --> O[페르소나 이미지 생성<br/>Google Gemini 1.5 Pro]
