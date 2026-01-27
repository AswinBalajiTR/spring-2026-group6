
# Capstone Proposal
## Autonomous Multi-Agent RAG System for Competitive Social Media Marketing Intelligence
### Proposed by: <Your Name>
#### Email: <Your Email>
#### Advisor: Amir Jafari
#### The George Washington University, Washington DC  
#### Data Science Program


## 1 Objective:  

            This capstone project will develop an autonomous multi-agent Retrieval-Augmented Generation (RAG) system that helps
            marketing teams in the consumer electronics (smartphone) industry monitor competitor activity, detect emerging trends,
            and generate platform-native marketing campaigns with competitive context.

            The system will continuously collect public competitor signals across major platforms (e.g., Instagram, X/Twitter,
            TikTok, YouTube, Reddit) and supporting sources (e.g., Meta Ad Library, Google Trends, tech news). Using RAG, the system
            will ground recommendations and generated content in retrieved evidence (recent competitor posts, high-performing examples,
            relevant trend signals, and product specs), reducing hallucinations and improving differentiation.

            The platform will orchestrate six specialized agents:
            1) Market Intelligence Agent: competitor monitoring, engagement analysis, campaign spike detection
            2) Trend Analysis Agent: emerging topics, consumer sentiment, seasonality, timing signals
            3) Creative Director Agent: campaign positioning and platform strategy brief
            4) Copywriting Agent: multi-variant captions, hooks, hashtags, CTAs adapted per platform
            5) Visual Designer Agent: creative briefs, layout guidance, and image/video prompt templates
            6) Strategy Orchestrator Agent: end-to-end coordination, output ranking, timing recommendations, and final packaging

            The outcome is a practical decision-support and content generation assistant that produces (a) competitor-aware insights,
            (b) clear positioning and gap analysis, and (c) complete campaign drafts (copy + creative briefs + posting schedule)
            that align with platform norms and brand goals.
            

![Figure 1: Example figure](2026_Spring_7.png)
*Figure 1: Caption*

## 2 Dataset:  

            The project will rely on publicly accessible datasets and sources (no restricted/FERPA data required). Data collection
            will be limited to public content and conducted with rate limiting and policy awareness.

            Core sources include:
            - X/Twitter: competitor posts, engagement metrics, timestamps (research/academic access if available)
            - YouTube Data API: brand and reviewer channels (titles, descriptions, view/like counts, transcripts where available)
            - Reddit API (PRAW): smartphone/tech subreddits for consumer sentiment and pain points
            - Instagram/TikTok: competitor post captions/metadata via compliant collection methods or approved dataset tools
            - Meta Ad Library: public ad creatives and copy for campaign signals
            - Google Trends (pytrends): keyword demand shifts and rising queries
            - Tech news via RSS/News sources: trend context and product cycle signals
            - GSMArena (or similar): structured product specifications for factual grounding

            Stored fields (examples):
            - caption/text, hashtags, timestamps, platform, brand/account
            - engagement metrics (as available), content type tags
            - derived features: sentiment score, topic cluster, posting frequency
            - embeddings + metadata for RAG retrieval (brand/platform/recency/topic)
            

## 3 Rationale:  

            Marketing teams face increasing difficulty producing differentiated content quickly while competitors launch campaigns
            across multiple platforms at high frequency. Manual monitoring is not scalable, and generic single-LLM tools often generate
            repetitive “AI-sounding” copy without competitive awareness.

            This project addresses that gap by building a multi-agent system grounded in real competitor and market signals via RAG.
            The system helps teams:
            - react faster to competitor launches and trend shifts,
            - identify underused messaging angles and creative gaps,
            - generate platform-native campaign drafts with evidence-backed reasoning.

            From a data science standpoint, the project integrates real-world pipelines (ETL), information retrieval (vector search),
            NLP (topic/sentiment), and agentic orchestration into an applied system that is evaluable, deployable, and directly relevant
            to modern AI product development.
            

## 4 Approach:  

            The project will be developed iteratively, starting with a minimum viable data pipeline and retrieval system, then layering
            multi-agent orchestration and quality controls.

            1) Data Ingestion + ETL:
               - Implement scheduled collectors for stable sources (YouTube, Reddit, RSS, Trends) first, then expand to additional
                 platforms where feasible.
               - Normalize metadata (brand, platform, timestamp, engagement fields) and compute derived signals (posting frequency,
                 engagement rate, sentiment, topic cluster).

            2) RAG Knowledge Base:
               - Generate text embeddings for posts/articles/specs and store in a vector database with searchable metadata filters
                 (brand, platform, recency, content type).
               - Use hybrid retrieval (dense + keyword) and recency/engagement boosting.

            3) Multi-Agent Orchestration:
               - Market Intelligence Agent: produces competitor summaries, alerts, and top-performing patterns.
               - Trend Analysis Agent: identifies rising topics, sentiment shifts, seasonality, and timing recommendations.
               - Creative Director Agent: synthesizes outputs into a campaign brief (positioning, tone, platform mix).
               - Copywriting Agent: generates multiple platform-specific caption variants + hashtags + CTAs.
               - Visual Designer Agent: outputs creative briefs (carousel/storyboard) and image/video prompt templates.
               - Strategy Orchestrator Agent: coordinates the workflow, ranks options, enforces constraints, and produces the final package.

            4) Validation + Safety + Quality:
               - Enforce citation-based generation: strategic claims and factual statements must be backed by retrieved sources.
               - Add guardrails to avoid unverifiable claims, unsafe targeting, and policy-violating recommendations.
               - Score outputs for clarity, differentiation, platform fit, and grounding.

            5) Evaluation:
               - Offline evaluation of retrieval relevance and summary quality.
               - Human evaluation of generated campaigns (marketing-quality rubric: clarity, brand fit, differentiation, usefulness).
               - Reliability checks: hallucination rate (unsupported claims), citation coverage, and consistency across agents.
            

## 5 Timeline:  

            Week 1:
              - Finalize scope: choose smartphone brands + platforms + success metrics (KPIs)
              - Confirm data access paths and compliance constraints

            Week 2: Weeks 1-2 - ideation, planning, and high level requirements gathering
              - Implement initial collectors (YouTube, Reddit, Google Trends, RSS)
              - Define schemas (Postgres tables + vector metadata) and ETL standards

            Week 3:
              - Build ETL cleaning + enrichment (sentiment, topics, posting frequency)
              - Create baseline competitor analytics outputs (top themes, post frequency)

            Week 4:
              - Embedding pipeline + vector DB setup
              - Baseline retrieval quality validation (precision checks)

            Week 5:
              - Implement Market Intelligence Agent + Trend Analysis Agent with RAG tools
              - Produce competitor summaries + trend briefs from retrieval

            Week 6:
              - Implement Creative Director + Copywriting Agents
              - Generate first end-to-end campaign drafts for 1–2 platforms

            Week 7: Weeks 3-7 - Prototype development and iteration round one
              - Add Visual Designer Agent (creative briefs + prompt templates)
              - Improve platform-specific formatting and constraints

            Week 8: Week 8 - ideation round 2 and develop final development plan
              - Add Strategy Orchestrator Agent with ranking + packaging
              - Add citation enforcement and output scoring

            Week 9:
              - Extend to more platforms/sources (as feasible)
              - Build lightweight UI (Streamlit) for competitor insights + campaign output

            Week 10:
              - Add alerting logic (campaign spike detection)
              - Improve retrieval boosting (recency/engagement weighting)

            Week 11:
              - Evaluation setup: rubrics, test prompts, baseline comparisons
              - Run retrieval + generation ablation tests (with/without RAG, with fewer agents)

            Week 12:
              - Pilot test with sample campaigns (internal testing / class demo)
              - Fix failure cases and harden system (stability, caching, rate limits)

            Week 13: Weeks 13-14 Project wrap up and continuity planning
              - Final documentation (README, architecture diagram, reproducibility)
              - Finalize demo and presentation

            Week 14:
              - Final capstone submission + live demo + handoff

            TOTAL: 14 weeks (one semester)

            KEY MILESTONES:
            - Week 2: Data sources confirmed + initial ingestion running
            - Week 4: Searchable RAG index online
            - Week 6: First end-to-end campaign generation working
            - Week 8: Full multi-agent orchestration + citation enforcement
            - Week 10: Alerts + UI integration
            - Week 12: Evaluation completed + system stabilized
            - Week 14: Final demo + documentation + submission

            DELIVERABLES BY WEEK 14:
            - Automated ingestion + ETL pipeline (reproducible)
            - Vector DB-backed RAG knowledge base with filters and boosting
            - Multi-agent orchestration producing competitor insights + campaigns
            - Campaign output package: positioning, captions (multi-variant), creative briefs, posting recommendations
            - Evaluation report (human rubric + reliability metrics)
            - Final demo + documentation + code repository
            


## 6 Expected Number Students:  

            RECOMMENDED: 2 students

            ROLE DISTRIBUTION FOR 2 STUDENTS:

            Student 1:
              - Responsibilities: Data ingestion/ETL, embeddings, vector DB, retrieval quality, analytics signals

            Student 2:
              - Responsibilities: Multi-agent orchestration, prompt/tool design, validation/guardrails, UI/demo, evaluation

            

## 7 Possible Issues:  

            TECHNICAL CHALLENGES AND SOLUTIONS:
            - Platform API restrictions / access variability:
              * Mitigation: prioritize stable sources first (YouTube, Reddit, RSS, Trends); keep collectors modular.
            - Noisy / inconsistent engagement metrics across platforms:
              * Mitigation: normalize by platform; use comparative trends rather than absolute counts.
            - LLM hallucinations or unverifiable marketing claims:
              * Mitigation: citation-based generation, hard constraints, and retrieval validation checks.
            - Scaling and cost:
              * Mitigation: start small (few brands/platforms), caching, batching embeddings, and selective model usage.

            RISK MITIGATION TIMELINE:
            - Weeks 1-2: finalize source list, confirm access, implement minimal viable ingestion
            - Weeks 3-4: lock schemas and stabilize ETL; validate retrieval before expanding scope
            - Weeks 5-6: ship early end-to-end MVP; add monitoring for failures
            - Weeks 7-8: add orchestration/ranking and grounding enforcement
            - Weeks 9-10: UI + alerting + scalability improvements
            - Weeks 11-12: run evaluation and address failure modes
            - Weeks 13-14: finalize documentation, demo hardening, and handoff plan
            


## Contact
- Author: Amir Jafari
- Email: [ajafari@gwu.edu](mailto:ajafari@gwu.edu)
- GitHub: [](https://github.com/)
