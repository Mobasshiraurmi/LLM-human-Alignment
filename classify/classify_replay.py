# Negotiation Behavior Classifier - Batch API Version v6, generalized over agent architecture.
#
# One classifier for any step5_replay.py --agent run: pass --agent {single,multi} and it
# reads/writes entirely inside replay/<agent>/ (clf_input.csv, batch_in/out.jsonl, batch_id.txt,
# results.csv), mirroring step5_replay.py's own per-agent layout. clf_input.csv is now built
# automatically from data/replay/<agent>/replay_unlabeled.csv (Role -> Sender rename) at --submit
# time -- previously a manual, undocumented step.
#
# Superseded classify_C1.py / classify_C4a.py / classify_C4d.py / classify_2A-C1.py (RAG
# conditions C4a/C4d are gone, and C1 vs 2A-C1 differed only in these file-path constants)
# are archived under archive/ -- see archive/README.md.
#
# Classification logic below is v6, unchanged from archive/classify_2A-C1.py (see that
# file's header for the v5->v6 fix history):
#
#   Fix A (9 errors): Tighten Progress Seeking.
#      Logistics/acknowledgement turns ("I'll see what I can do", "Sure, lets
#      see", "Certainly, one moment") were firing Progress Seeking. Progress
#      Seeking requires introducing a NEW conditional or mechanism, not just
#      signalling willingness. Negative examples added.
#
#   Fix B (5 errors): Screen/interface observations -> Neutral.
#      Turns describing what the participant sees on their screen ("on my screen
#      i see 60k", "Stock options goes up to 20,000", "it stayed the same for
#      me most of the rounds") were coded as Share Information. These are
#      interface observations, not personal disclosures.
#
#   Fix C (3 errors): Preference statements != offers.
#      "I want 175K base salary...", "I prefer a $120,000 salary..." state
#      desires, not proposals to the other party. = Share Information, not
#      Make integrative(multi-issue) offers.
#
#   Fix D (18 errors): Relationship building internal inconsistency.
#      14 of 18 errors had the model correctly predicting subtype as
#      Relationship building statement but outputting Neutral as behavior --
#      an internal contradiction. Fixed by making behavior derivation explicit
#      in the output format: behavior must be consistent with subtype.
#      Also added 4 edge cases: hopeful/polite closings that are warm but
#      brief ("Hope that we can reach a double-win outcome", "it was nice
#      working with you") are Relationship building, not Neutral.
#
#   Fix E (9 errors): Collaborative Statement -> Neutral/Distributive.
#      "seems that all in the middle works for both of us", "we seem to be in
#      agreement" -> Neutral (bare confirmation). "I have changed it for our
#      benefit", "i offered you the highest numbers possible" -> Distributive
#      (persuasive framing). Added negative examples for both directions.
#
#   Scoring fix: normalize double spaces in subtype labels before kappa
#      computation. "Relationship  building  statement" (CSV) vs
#      "Relationship building statement" (model output) caused 0% apparent
#      accuracy on that subtype despite 67% true accuracy.
#
# Usage:
#   python classify/classify_replay.py --agent single --submit
#   python classify/classify_replay.py --agent single --check
#   python classify/classify_replay.py --agent single --parse
#   python classify/classify_replay.py --agent multi  --submit / --check / --parse
import os
import sys
import json
import hashlib
import argparse
import time
import pandas as pd
from pathlib import Path
from openai import OpenAI
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MODEL          = "gpt-5.4"
N_CONTEXT      = 5
TEMPERATURE    = 0.0

client = OpenAI(api_key=OPENAI_API_KEY)

SUBTYPE_TO_BEHAVIOR = {
    "Neutral": "Neutral",
    "Agreement / finalize remaining issue": "Neutral",
    "Ask(open-ended) questions": "Integrative",
    "Active listening": "Integrative",
    "Share Information": "Integrative",
    "Make integrative(multi-issue) offers": "Integrative",
    "Collaborative Statement": "Integrative",
    "Progress Seeking Statement": "Integrative",
    "Relationship building statement": "Integrative",
    "Ask positionbased question": "Distributive",
    "Discuss one issue at a time": "Distributive",
    "Make single issue offers": "Distributive",
    "Substantiate position or refer to bottom line": "Distributive",
    "Use of force/misrepresentation": "Distributive",
}

# ---------------------------------------------------------------------------
# FRAMEWORK -- domain-general. v6: Fix A (PS tightened), Fix B (screen obs),
# Fix C (preference statements), Fix D (RB edge cases), Fix E (CS boundaries).
# ---------------------------------------------------------------------------
FRAMEWORK = """NEGOTIATION BEHAVIOR FRAMEWORK (Heunis et al. 2024)

You are coding ONE speaking turn from a multi-issue negotiation between two parties.
The negotiation concerns several divisible issues (the specific items vary by
scenario). Classify the turn into exactly ONE subtype. Subtypes roll up to three
behavior categories: Neutral, Integrative (cooperative), Distributive (competitive).

Code based on the negotiation MOVE the turn performs, never on the specific items
or numbers mentioned.

=== NEUTRAL (check this FIRST) ===
A turn is Neutral when it performs NO strategic negotiation move. Default for any
turn that does not clearly fit an Integrative or Distributive subtype. Includes:
- Greetings, introductions, small talk, sign-offs.
- Bare confirmations with no content: "yes", "ok", "great", "done", "agreed",
  "sounds good", "I understand", "deal". Neutral EVEN mid-negotiation.
- Interface / logistics / mechanics: did you submit?, I sent the offer,
  are you seeing it on your end?, click submit, what round are we on,
  Before we send it in one more run through, Certainly one moment as I
  calculate the offer, I'll submit this intermediate offer for this round,
  Hey I'm going to send you some figures over in a few, Let me try that.
- Screen/interface observations: turns describing what the participant sees
  on their interface rather than disclosing personal information -- "on my
  screen i see 60k as the lowest salary, 90k as the highest", "Stock options
  goes up to 20,000", "it stayed the same for me most of the rounds",
  "on my end the lowest bonus is 0%, second is 3, highest is 5",
  "mine goes up to 5k". These are observations about the system, not
  personal disclosures. Code Neutral.
- Reactions or observations with no proposal/argument: that is quite low,
  this is fast, typo corrections, let me look.
- Generic cooperative openers with no actual proposal or disclosure: let's be
  fair, let's meet in the middle, we need a win-win outcome are Neutral
  UNLESS paired with a concrete proposal, trade-off, or substantive disclosure.
- Agreement / finalize remaining issue: explicitly closing the WHOLE deal.
  (Only for genuine final acceptance, not a mid-stream ok.)

=== INTEGRATIVE (cooperative -- expands or explores value) ===
- Ask(open-ended) questions: gathering information about the OTHER party's
  interests, priorities, or preferences; broad/exploratory questions. Questions
  asking for a RANGE are open-ended, not positional. Confirmatory questions
  ("are you agreeing to X?", "great so we do have a deal?") = Ask(open-ended),
  not Neutral.
- Active listening: summarising or PARAPHRASING what the other party said, or
  a follow-up to clarify their words. Requires engaging with their content.
- Share Information: honestly disclosing OWN interests, priorities, constraints,
  or factual context to build mutual understanding. Includes:
  (a) Disclosing prior salary, company culture, BATNA, personal situation.
  (b) Stating own PREFERENCES or DESIRES about the package: "I want 175K base
      salary 30 vacation days and 10% stock options", "I prefer a $120,000
      salary with 20% bonus", "My preferences are 110K salary, 10% bonus".
      These state what the speaker wants, not a proposal to the other party.
      Share Information, not Make integrative(multi-issue) offers.
  Key test for Share Information vs Substantiate: is the speaker explaining
  their situation, or using it to resist? Explanation = Share Information.
  Constraint invoked to block movement = Substantiate.
- Make integrative(multi-issue) offers: a PROPOSAL to the other party spanning
  MULTIPLE issues, or adjusting one element of an ongoing package. Requires
  actually proposing -- not just stating preferences. Includes multi-value
  strings as direct offers ("80,000, 10%, 10 vacation", "sending offer of
  100000 10 bonus 10000 stock options 15 vacation days"). Also includes turns
  that address one issue BUT reference others or acknowledge the full package.
- Collaborative Statement: appeals to MUTUAL interest or shared ground grounded
  in the actual negotiation. Includes warm agreement statements that acknowledge
  convergence: "ok i think i can do that exactly too..", "Everything is what
  you want", "we seem to be in agreement", "seems that all in the middle works
  for both of us". NOT "we seem to be in agreement" said as a bare confirmation
  with no elaboration -- context determines whether it is Collaborative or
  closing/Neutral. NOT turns expressing firmness framed cooperatively: "I have
  changed it for our benefit", "i offered you the highest numbers possible"
  these are Distributive (Substantiate) despite cooperative framing.
- Progress Seeking Statement: introduces a NEW path, mechanism, or CONDITIONAL
  to move things forward. The test: is the speaker proposing something new?
  YES (Progress Seeking): "if you do X, I can do Y", "what if we split the
  difference?", "agree on a salary of 110k?", "Is there anything I can do
  differently?", "ok. I will try another option in the next round to see
  whether you could reach", "Let's see if we calculate it down the middle".
  NO (Neutral, not Progress Seeking): "I'll see what I can do", "Sure, lets
  see", "Certainly, one moment", "Surely, I will consider your request",
  "Let me try that", "we will make" -- these are acknowledgements or logistics,
  not new proposals. The forward motion must be substantive, not just polite.
- Relationship building statement: deliberate warmth, empathy, humor, gratitude,
  enthusiasm, or rapport beyond a routine greeting. INCLUDES:
  (a) Clear warmth/humor: "wow, that's awesome, thank you!!", "hahaha commission
      only!", "Welcome to the company!", "Oh haha I was like 'bold move'",
      "I'm thrilled to be here", "we'd love to have you on board".
  (b) Hopeful or warm closings that express goodwill: "Hope that we can reach a
      double-win outcome", "it was nice working with you", "Yes it went well",
      "Thats alright" (said warmly at end of negotiation), "I appreciate you",
      "Thank you so much for the offer. I really would appreciate a good one!",
      "great, thank you, i hope it doesn't lower your score too much",
      "i don't want this to be awful for you either".
  Test: does the turn express genuine warmth or goodwill toward the other party,
  even briefly? If yes, Relationship building, not Neutral.

=== DISTRIBUTIVE (competitive -- claims value) ===
- Ask positionbased question: asks for a specific position/bottom line on ONE
  issue. Asking for a RANGE = Ask(open-ended) questions.
- Discuss one issue at a time: deliberately isolates to ONE issue with explicit
  isolation language.
- Make single issue offers: an offer on ONE issue with no package reference.
  "120000 please", "Please make it 130000", "Let us settle with 100,000",
  "I can do 25 days", "100,000 salary", "I can do the 15 vacation days though".
- Substantiate position or refer to bottom line: persuasive argument or
  resistance signal. Three forms:
  (a) Explicit limits: "that's my limit", "I already gave up a lot", "this is
      my best offer", "that's below my backup so I can't accept".
  (b) Budget/constraint used to justify why the offer cannot improve: "budget
      is tight", "we can't break the bank", "Bonuses, Options, and vacation
      days don't harm the company" (implying salary cannot move). Resistance
      intent does not need to be explicit.
  (c) Prior experience as leverage: "That sounds interesting but I was taking
      more than that in my previous company".
  (d) Cooperative-sounding but actually firm: "I have changed it for our
      benefit", "i offered you the highest numbers possible", "I think you will
      like this offer", "I think you can work with this offer" -- these express
      firmness with cooperative framing. Substantiate, not Collaborative.
  NOT Substantiate: constraint mentioned alongside a concrete offer as neutral
  context with no resistance framing.
- Use of force/misrepresentation: threats, ultimatums, time pressure, deception,
  or citing a COMPETING outside offer as leverage.

=== BOUNDARY RULES ===
RULE 1 -- STANCE OVER ISSUE COUNT. Firm/positional framing = Distributive even
if multiple issues are listed. Integrative offers leave room; distributive ones
draw a line.

RULE 2 -- SHARE INFORMATION vs SUBSTANTIATE. Explanation alongside an offer or
as neutral context = Share Information. Constraint invoked to justify holding a
position = Substantiate. Preference statements ("I want X", "I prefer X") =
Share Information regardless of whether they contain numbers.

RULE 3 -- SINGLE vs MULTI-ISSUE OFFER. Turn acknowledging the package or naming
multiple values = Make integrative(multi-issue) offers. Turn focusing on one item
only with no package reference = Make single issue offers.

RULE 4 -- BEHAVIOR MUST MATCH SUBTYPE. Every subtype belongs to exactly one
behavior: Neutral subtypes -> Neutral, Integrative subtypes -> Integrative,
Distributive subtypes -> Distributive. If you classify a turn as Relationship
building statement, the behavior MUST be Integrative. If you classify it as
Substantiate, the behavior MUST be Distributive. Never output a behavior that
contradicts your chosen subtype."""

# ---------------------------------------------------------------------------
# DECISION PROCEDURE — v6: Neutral-first, warmth rescue, explicit subtype->
# behavior consistency enforcement (Fix D).
# ---------------------------------------------------------------------------
DECISION = """DECISION PROCEDURE (apply in order):
1. Is the turn a greeting, bare confirmation, logistics/interface comment,
   screen/system observation, or contentless reaction with NO strategic content?
   → NEUTRAL. Stop.
2. Does the turn express deliberate warmth, humor, enthusiasm, gratitude, or
   rapport beyond minimal social convention — including hopeful/warm closings?
   → INTEGRATIVE / Relationship building statement. Stop.
3. Is the framing firm/positional, an ultimatum, a bottom-line/limit reference,
   a persuasive justification for own position, or a competing-offer threat?
   → DISTRIBUTIVE (pick the matching subtype). Stop.
4. Otherwise it explores, discloses, proposes, appeals to mutual interest, or
   seeks a new path → INTEGRATIVE (pick the subtype).

CRITICAL — BEHAVIOR MUST MATCH SUBTYPE (Rule 4):
After choosing a subtype, derive behavior from it automatically:
- Neutral subtype (Neutral, Agreement/finalize) → behavior = Neutral
- Any Integrative subtype → behavior = Integrative
- Any Distributive subtype → behavior = Distributive
Never output a behavior label that contradicts your chosen subtype. If you
chose "Relationship building statement" the behavior must be "Integrative",
not "Neutral". If you chose "Substantiate position" behavior must be
"Distributive", not "Integrative".

When torn between Neutral and Integrative for a short turn: prefer NEUTRAL
unless it expresses warmth (step 2) or contains a clear strategic move.
When torn between Integrative and Distributive: let RULE 1 (STANCE) decide."""

# ---------------------------------------------------------------------------
# FEW-SHOT — domain-neutral. Examples chosen so the STRUCTURAL cue generalizes
# beyond salary. Each shows the move, not the merchandise.
# ---------------------------------------------------------------------------
FEWSHOT = """EXAMPLES (the cue is the MOVE, not the items):

--- Neutral ---
"hello there, nice to meet you" -> Neutral
"yes." -> Neutral
"great" -> Neutral
"okay sounds good :)" -> Neutral
"I understand" -> Neutral
"did you submit your offer?" -> Neutral                (interface/logistics)
"I sent it on my end" -> Neutral                       (interface/logistics)
"Before we send it in, one more run through" -> Neutral  (logistics, not PS)
"Certainly, one moment as I calculate the offer" -> Neutral  (logistics, not PS)
"I'll see what I can do" -> Neutral                    (polite ack, not PS -- no new mechanism)
"Sure, lets see" -> Neutral                            (polite ack, not PS)
"Surely, I will consider your request" -> Neutral      (polite ack, not PS)
"Let me try that" -> Neutral                           (logistics)
"that is quite low" -> Neutral                         (reaction, no proposal)
"let's be fair" -> Neutral                             (generic opener, no proposal)
"on my screen i see 60k as the lowest salary, 90k as the highest" -> Neutral  (screen observation)
"Stock options goes up to 20,000" -> Neutral           (screen observation, not Share Info)
"it stayed the same for me most of the rounds" -> Neutral  (screen observation)
"on my end the lowest bonus is 0%, second is 3, highest is 5" -> Neutral  (screen observation)
"mine goes up to 5k" -> Neutral                        (screen observation)
"Ready to begin discussion of Employee benefits package" -> Neutral  (opener/logistics)
"Very well, I am preparing an offer for you" -> Neutral  (logistics)
"May I get your name" -> Neutral                       (small talk)

--- Closing ---
"ok I'm good with everything, deal" -> Agreement / finalize remaining issue

--- Relationship building (warmth/humor/goodwill -- even brief) ---
"wow, that's awesome, thank you!!" -> Relationship building statement
"hahaha commission only!" -> Relationship building statement
"Welcome to the company!" -> Relationship building statement
"Oh haha I was like 'bold move'" -> Relationship building statement
"I'm thrilled to be here and excited to hear your offer" -> Relationship building statement
"we'd genuinely love to have you on board!" -> Relationship building statement
"it's my pleasure, I hope you will like this offer" -> Relationship building statement
"Hope that we can reach a double-win outcome" -> Relationship building statement  (hopeful goodwill)
"it was nice working with you" -> Relationship building statement  (warm closing)
"I appreciate you" -> Relationship building statement
"great, thank you, i hope it doesn't lower your score too much" -> Relationship building statement
"i don't want this to be awful for you either" -> Relationship building statement  (empathy)
"Again I really appreciate everything! Let's hope this would be the right one" -> Relationship building statement

--- Collaborative Statement ---
"ok i think i can do that exactly too.." -> Collaborative Statement
"Everything is what you want." -> Collaborative Statement
"Oh, is that so? Well, I don't see why not. We should aim to hire the best!" -> Collaborative Statement
"seems that all in the middle works for both of us" -> Collaborative Statement
"we seem to be in agreement" -> Collaborative Statement
"let's find something that works for both of us" -> Collaborative Statement
NOT Collaborative: "I have changed it for our benefit" -> Substantiate  (firmness with cooperative framing)
NOT Collaborative: "i offered you the highest numbers possible" -> Substantiate  (firmness)
NOT Collaborative: "I think you will like this offer" -> Substantiate  (pressure framing)

--- Progress Seeking (NEW mechanism or conditional -- not just polite willingness) ---
"agree on a salary of 110k?" -> Progress Seeking Statement
"Is there anything that I can do differently?" -> Progress Seeking Statement
"ok. I will try another option in the next round to see whether you could reach" -> Progress Seeking Statement
"Let's see if we calculate it down the middle if it puts us both in profit" -> Progress Seeking Statement
"what if you take X and I take Y this round?" -> Progress Seeking Statement
"if you do that, I can move on the other issue" -> Progress Seeking Statement
NOT Progress Seeking: "I'll see what I can do" -> Neutral  (polite ack, no new mechanism)
NOT Progress Seeking: "Sure, lets see" -> Neutral
NOT Progress Seeking: "Certainly, one moment" -> Neutral

--- Ask(open-ended) questions ---
"what matters most to you here?" -> Ask(open-ended) questions
"what range of salary would you feel comfortable with?" -> Ask(open-ended) questions
"great so we do have a deal?" -> Ask(open-ended) questions
"are you agreeing to $110, 15% bonus, $10k stocks and 15 vacation days?" -> Ask(open-ended) questions

--- Active listening ---
"so if I'm hearing you right, you'd take less on the first item for more on the second?" -> Active listening
"OH I see" -> Active listening

--- Share Information (disclosures AND preference statements) ---
"honestly the two things I care about most are A and B" -> Share Information
"I recently left a company where I had 24 vacation days and 10k in options" -> Share Information
"I want 175K base salary 30 vacation days and 10% stock options" -> Share Information  (stating preferences, not proposing)
"I prefer a $120,000 salary with 20% bonus, $15,000 stock options and 25 vacation days" -> Share Information  (preference, not offer)
"My preferences are 110K salary, 10% bonus, 10k stock options, with 15 vacation days" -> Share Information  (preference, not offer)
"i think it is a fair mix of options" -> Share Information  (disclosing own view)
"I feel that stocks are an investment in my future" -> Share Information

--- Make integrative(multi-issue) offers (PROPOSALS to the other party) ---
"I could give you more on A if you come down on B and C" -> Make integrative(multi-issue) offers
"80,000, 10%, 10 vacation" -> Make integrative(multi-issue) offers  (multi-value offer)
"sending offer of 100000 10 bonus 10000 stock options 15 vacation days" -> Make integrative(multi-issue) offers
"how about 20k stock?" [mid-package exchange] -> Make integrative(multi-issue) offers
"No, I am thinking 110k salary, is that too much for you? 10% bonus. 15k stock." -> Make integrative(multi-issue) offers
"ok, we can work to increase your vacation days and bonuses but $120 is for 10+ years experience" -> Make integrative(multi-issue) offers

--- Make single issue offers ---
"120000 please" -> Make single issue offers
"Please make it 130000" -> Make single issue offers
"I can do 25 days" -> Make single issue offers
"100,000 salary" -> Make single issue offers
"I love the offer, I will have to push again for the 110k salary, everything else is good." -> Make single issue offers

--- Substantiate position or refer to bottom line ---
"this is my final offer across the board" -> Substantiate position or refer to bottom line
"that's below my backup, I can't accept" -> Substantiate position or refer to bottom line
"budget is tight" -> Substantiate position or refer to bottom line
"Bonuses, Options, and vacation days don't harm the company." -> Substantiate position or refer to bottom line
"That sounds interesting but I was taking more than that in my previous company" -> Substantiate position or refer to bottom line
"I have changed it for our benefit" -> Substantiate position or refer to bottom line  (cooperative framing, but firmly holding position)
"i offered you the highest numbers possible" -> Substantiate position or refer to bottom line
"I think you will like this offer" -> Substantiate position or refer to bottom line  (soft pressure)
"I have another offer on the table that's better" -> Use of force/misrepresentation

--- Boundary cases ---
"that's significantly less than my backup...I would do 110, 15, 15k, 15 vaca" -> Make integrative(multi-issue) offers  (offer wins)
"my backup is 50 points on my screen" -> Share Information  (explaining, not resisting)
"that puts me under my backup so I can't accept" -> Substantiate position or refer to bottom line
"We have a tight budget right now" (standalone) -> Substantiate position or refer to bottom line
"We have included some bonuses on holidays which brings the basic salary to 90000" -> Make integrative(multi-issue) offers"""


# ---------------------------------------------------------------------------
# PREMISE — domain-general. Describes the STRUCTURE of the experiment without
# naming the goods. Optionally inject a one-line scenario hint at runtime.
# ---------------------------------------------------------------------------
PREMISE = """PREMISE: A multi-issue negotiation between two parties over several divisible
issues. Each side has a private fallback (BATNA). Partway through, one side's BATNA
is perturbed, which can shift strategy. Most turns are part of an ongoing package
negotiation, so read each turn in the context of the turns before it. The specific
items being negotiated vary by scenario and are NOT relevant to the coding — code
the negotiation move, not the subject matter."""

SYSTEM_PROMPT = f"""You are an expert negotiation researcher coding speaking turns with the Heunis et al. (2024) framework. You code the negotiation MOVE a turn performs, independent of the domain or the specific items/numbers involved.

{FRAMEWORK}

{DECISION}

{FEWSHOT}

Classify ONLY the turn marked >>>. Use the conversation history as context — it is
essential for short turns and for single-issue mentions inside a multi-issue exchange.

Respond ONLY in this JSON format, no markdown:
{{"subtype": "<exact subtype name>", "reasoning": "<one sentence naming the move and citing the rule/definition>"}}"""


def paths(agent: str, out_dir: str = "data/replay") -> dict:
    rdir = f"{out_dir}/{agent}"
    return {
        "run_dir": rdir,
        "unlabeled_csv": f"{rdir}/replay_unlabeled.csv",
        "input_csv": f"{rdir}/clf_input.csv",
        "output_csv": f"{rdir}/results.csv",
        "batch_in": f"{rdir}/batch_in.jsonl",
        "batch_out": f"{rdir}/batch_out.jsonl",
        "batch_id": f"{rdir}/batch_id.txt",
    }


def build_clf_input(p: dict) -> pd.DataFrame:
    """data/replay/<agent>/replay_unlabeled.csv (written by step5_replay.py's export()) ->
    replay/<agent>/clf_input.csv, in the schema this classifier expects. Regenerated every
    --submit, so it always reflects the current replay_unlabeled.csv.
    """
    df = pd.read_csv(p["unlabeled_csv"])
    df = df.rename(columns={"Role": "Sender"})[["GameId", "Sender", "Text", "Phase", "ProlificId", "Round"]]
    df.to_csv(p["input_csv"], index=False)
    print(f"wrote {p['input_csv']} ({len(df)} rows, from {p['unlabeled_csv']})")
    return df


def make_request_key(idx, text, game_id):
    raw = f"v6|{idx}|{game_id}|{text}|{MODEL}"
    return hashlib.sha1(raw.encode()).hexdigest()


def build_user_message(df, idx, scenario_hint=None):
    row = df.iloc[idx]
    game = row["GameId"]
    context_lines = []
    for j in range(max(0, idx - N_CONTEXT), idx):
        r = df.iloc[j]
        if r["GameId"] == game:
            context_lines.append(f"  {r['Sender']}: {r['Text']}")
    context_block = "\n".join(context_lines) if context_lines else "  (start of conversation)"

    # Optional scenario line lets the SAME prompt run on salary OR antique-dealer
    # data: pass a neutral one-liner, e.g. "Scenario: two dealers dividing items."
    scenario_block = f"\n{scenario_hint}\n" if scenario_hint else ""

    # Phase is given by the experimental design; including it is allowed (mentor
    # confirmed) but it must NOT change the subtype — it only flags that a BATNA
    # shift may have occurred. Comment out the next line to ablate phase.
    phase = row.get("Phase", "")
    phase_block = f"(phase: {phase})\n" if phase else ""

    return f"""{PREMISE}
{scenario_block}
CONVERSATION SO FAR:
{context_block}

{phase_block}>>> CLASSIFY THIS TURN ({row['Sender']}): "{row['Text']}"
"""


def build_and_submit(p: dict, df: pd.DataFrame, scenario_hint=None):
    print(f"Building v6 batch for {len(df)} turns...")
    requests = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        key = make_request_key(idx, row["Text"], row["GameId"])
        requests.append({
            "custom_id": key,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(df, idx, scenario_hint)},
                ],
                "temperature": TEMPERATURE,
                "max_completion_tokens": 200,
                "response_format": {"type": "json_object"},
            },
        })

    with open(p["batch_in"], "w", encoding="utf-8") as f:
        for r in requests:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(requests)} requests to {p['batch_in']}")

    up = client.files.create(file=open(p["batch_in"], "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=up.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    Path(p["batch_id"]).write_text(batch.id)
    print(f"Submitted batch {batch.id} (status: {batch.status})")


def run_sync(p: dict, df: pd.DataFrame, scenario_hint=None):
    """Real-time classification -- same SYSTEM_PROMPT/build_user_message/
    SUBTYPE_TO_BEHAVIOR as build_and_submit, called directly in a resumable loop
    instead of through the Batch API. Use when the Batch API is stalled (no error,
    just stuck at 0 progress for a long time -- seen twice on classify_freerun.py's
    runs). ~2x the per-token cost of batch (no 50% discount), but nothing to get
    stuck in a queue for -- entirely under our own control, same resumability
    pattern as step5_replay.py/step2_generate.py's generation loops.

    Writes p['output_csv'] directly (same schema build_and_submit/parse_and_score
    produces), so report_replay.py doesn't care which path was used. Ported
    verbatim from classify_freerun.py's run_sync.
    """
    out_path = p["output_csv"]
    df = df.copy()
    df["pred_subtype"] = None
    df["pred_behavior"] = None
    df["pred_reasoning"] = None

    done = 0
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        if len(prev) == len(df):
            has_pred = prev["pred_behavior"].notna()
            df.loc[has_pred, ["pred_subtype", "pred_behavior", "pred_reasoning"]] = \
                prev.loc[has_pred, ["pred_subtype", "pred_behavior", "pred_reasoning"]].values
            done = int(has_pred.sum())
    if done:
        print(f"resuming: {done}/{len(df)} already classified")

    for idx in range(len(df)):
        if pd.notna(df.at[idx, "pred_behavior"]):
            continue
        user_msg = build_user_message(df, idx, scenario_hint)
        subtype, reasoning = "Neutral", ""
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model=MODEL, temperature=TEMPERATURE,
                    max_completion_tokens=200,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": user_msg}],
                )
                parsed = json.loads(r.choices[0].message.content)
                subtype = parsed.get("subtype", "Neutral").strip()
                reasoning = parsed.get("reasoning", "")
                break
            except Exception as exc:
                if attempt == 3:
                    print(f"  FAILED idx={idx}: {exc!r} -- defaulting to Neutral")
                else:
                    time.sleep(2 ** attempt)
        df.at[idx, "pred_subtype"] = subtype
        df.at[idx, "pred_behavior"] = SUBTYPE_TO_BEHAVIOR.get(subtype, "Neutral")
        df.at[idx, "pred_reasoning"] = reasoning

        if (idx + 1) % 25 == 0 or idx == len(df) - 1:
            df.to_csv(out_path, index=False)
            print(f"  {idx + 1}/{len(df)}", flush=True)

    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(df["pred_behavior"].value_counts().to_string())


def check(p: dict):
    bid = Path(p["batch_id"]).read_text().strip()
    b = client.batches.retrieve(bid)
    print(f"Batch {bid}: {b.status}")
    if b.request_counts:
        print(f"  completed={b.request_counts.completed} "
              f"failed={b.request_counts.failed} total={b.request_counts.total}")
    return b


def parse_and_score(p: dict):
    b = check(p)
    if b.status != "completed":
        print("Batch not complete yet.")
        return
    out = client.files.content(b.output_file_id).text
    with open(p["batch_out"], "w", encoding="utf-8") as f:
        f.write(out)

    df = pd.read_csv(p["input_csv"])
    key_to_idx = {make_request_key(i, df.iloc[i]["Text"], df.iloc[i]["GameId"]): i
                  for i in range(len(df))}

    df["pred_subtype"] = None
    df["pred_behavior"] = None
    df["pred_reasoning"] = None

    for line in out.strip().splitlines():
        obj = json.loads(line)
        cid = obj["custom_id"]
        idx = key_to_idx.get(cid)
        if idx is None:
            continue
        try:
            content = obj["response"]["body"]["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            subtype = parsed.get("subtype", "Neutral").strip()
        except Exception:
            subtype = "Neutral"
        df.at[idx, "pred_subtype"] = subtype
        df.at[idx, "pred_behavior"] = SUBTYPE_TO_BEHAVIOR.get(subtype, "Neutral")

    df.to_csv(p["output_csv"], index=False)
    print(f"Wrote {p['output_csv']}")

    # Normalize spacing in subtype labels before scoring
    # (CSV uses double spaces in "Relationship  building  statement"; model uses single)
    import re
    def norm(s):
        return re.sub(r' +', ' ', str(s).strip()) if pd.notna(s) else s

    df["pred_subtype_norm"] = df["pred_subtype"].apply(norm)

    # Score against human behavior labels, if this input happens to carry gold labels
    # (it won't for model-turn classification -- see report_replay.py for the actual
    # human-vs-model alignment comparison).
    truth_col = ("My_Behavior" if "My_Behavior" in df.columns
                 else "Mentor_Behavior" if "Mentor_Behavior" in df.columns
                 else None)
    if truth_col is None:
        print("\nNo gold-label column -- skipping scoring "
              "(expected for model-turn classification).")
        print(df["pred_behavior"].value_counts().to_string())
        return
    valid = df[df["pred_behavior"].notna()]
    y_true = valid[truth_col]
    y_pred = valid["pred_behavior"]
    print(f"\nLLM vs {truth_col}  n={len(valid)}")
    print(f"Behavior Cohen's kappa: {cohen_kappa_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred,
          labels=["Neutral", "Integrative", "Distributive"], zero_division=0))
    print("Confusion matrix [Neutral, Integrative, Distributive]:")
    print(confusion_matrix(y_true, y_pred,
          labels=["Neutral", "Integrative", "Distributive"]))

    # Score against human subtype labels (normalized)
    subtype_col = "My_Subtype" if "My_Subtype" in df.columns else "BehaveSubtype"
    if subtype_col in df.columns:
        valid_s = df[df["pred_subtype_norm"].notna()].copy()
        valid_s["subtype_truth_norm"] = valid_s[subtype_col].apply(norm)
        print(f"\nSubtype Cohen's kappa (LLM vs {subtype_col}, normalized): "
              f"{cohen_kappa_score(valid_s['subtype_truth_norm'], valid_s['pred_subtype_norm']):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=["single", "multi", "single_personality", "single_rag", "single_claude-sonnet-4.6", "single_gemini-2.5-pro", "single_personality_claude-sonnet-4.6", "single_personality_gemini-2.5-pro", "multi_claude-sonnet-4.6", "multi_gemini-2.5-pro"],
                    help="which step5_replay.py run to classify (replay/<agent>/).")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--parse", action="store_true")
    ap.add_argument("--sync", action="store_true",
                    help="Classify in real time instead of via the Batch API -- use when "
                         "the batch is stalled. ~2x cost, no queue to get stuck in, resumable. "
                         "Writes results.csv directly; no --check/--parse needed afterward.")
    ap.add_argument("--scenario", type=str, default=None,
                    help="Optional one-line scenario hint, e.g. "
                         "'Scenario: two antique dealers dividing items.'")
    ap.add_argument("--out-dir", default="data/replay",
                    help="base directory to classify (default: data/replay, the shipped results). "
                         "Point this elsewhere (e.g. run_pipeline.py --full's default) to classify "
                         "a fresh run without touching the committed data.")
    args = ap.parse_args()
    p = paths(args.agent, args.out_dir)

    if args.sync:
        df = build_clf_input(p)
        run_sync(p, df, scenario_hint=args.scenario)
    elif args.submit:
        df = build_clf_input(p)
        build_and_submit(p, df, scenario_hint=args.scenario)
    elif args.check:
        check(p)
    elif args.parse:
        parse_and_score(p)
    else:
        print("Use --submit / --check / --parse, or --sync for real-time classification")
