"""
Sample Data Generator
======================

Generates realistic synthetic fake and real news articles for training
and evaluating the Fake News Intelligence System.

Real articles mimic formal journalistic writing with proper attribution,
specific facts, and neutral tone. Fake articles use sensationalist
language, vague sourcing, emotional manipulation, and clickbait patterns.

Usage:
    python -m data.generate_sample_data
    # or
    python data/generate_sample_data.py

Output:
    data/sample/sample_data.csv
"""

import os
import random
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ====================================================================== #
#  REAL article templates                                                 #
# ====================================================================== #

_REAL_INTROS = [
    "Reuters reported on {date} that",
    "According to officials familiar with the matter,",
    "The Department of {dept} confirmed on {date} that",
    "In a statement released {date}, the {org} announced that",
    "Government data released on {date} shows that",
    "A peer-reviewed study published in {journal} found that",
    "During a press conference on {date}, {person} stated that",
    "Official figures from the {org} indicate that",
    "The Associated Press has confirmed that",
    "According to a report by the {org},",
    "Newly released documents show that",
    "In testimony before Congress on {date},",
    "Data from the Bureau of {dept} reveals that",
    "A spokesperson for the {org} told reporters that",
    "In an exclusive interview with {outlet},",
    "Federal investigators announced on {date} that",
    "The {org} released its quarterly report showing that",
    "According to census data released {date},",
    "An independent audit conducted by {org} found that",
    "Court documents filed on {date} reveal that",
]

_REAL_BODIES_POLITICS = [
    'the bipartisan infrastructure bill has secured {num} votes in the Senate, paving the way for $1.2 trillion in spending over the next decade. "{quote}," said Senator {person}.',
    "new regulations on data privacy will take effect next quarter, requiring companies with over 500 employees to appoint a chief privacy officer. The legislation passed with a {num}-vote margin.",
    'the administration\'s proposed budget allocates ${num} billion to education funding, a 12% increase from the previous fiscal year. "{quote}," the education secretary noted.',
    "diplomatic talks between the two nations have resulted in a preliminary agreement on tariff reductions affecting approximately ${num} billion in bilateral trade.",
    'voter registration in the state increased by {num}% compared to the same period last year, according to the Secretary of State\'s office. "{quote}," the official added.',
    'the newly appointed ambassador presented credentials to the foreign ministry on {date}. "{quote}," the State Department confirmed in a written statement.',
    "a bipartisan committee recommended {num} policy changes to strengthen election security infrastructure ahead of upcoming midterm elections.",
    'the governor signed an executive order allocating ${num} million to wildfire prevention after consultation with emergency management officials. "{quote}," she said.',
]

_REAL_BODIES_BUSINESS = [
    "quarterly earnings exceeded analyst expectations, with revenue reaching ${num} billion, up 8% year-over-year. The company attributed growth to strong demand in the cloud computing segment.",
    "the Federal Reserve held interest rates steady at {num}%, citing mixed signals in employment data. Economists surveyed by Bloomberg had predicted the decision.",
    'the merger between {org} and {org2} received regulatory approval on {date}, creating a combined entity valued at approximately ${num} billion. "{quote}," the CEO announced.',
    "unemployment claims fell to {num},000 last week, the lowest level since March, according to Labor Department figures released Thursday.",
    'consumer spending rose {num}% in the third quarter, driven by increased demand for services. "{quote}," noted the chief economist at {org}.',
    "supply chain disruptions in the semiconductor industry are expected to ease by Q3, according to a joint report by {org} and industry analysts.",
    'the initial public offering priced shares at ${num}, valuing the company at approximately $14 billion. "{quote}," the CFO told analysts during the roadshow.',
    "trade volume at the port of Los Angeles increased {num}% in October compared to the prior year, reflecting improved global shipping logistics.",
]

_REAL_BODIES_TECH = [
    'researchers at {org} demonstrated a quantum computing breakthrough, achieving {num}-qubit stable operation for the first time. The results were published in Nature on {date}. "{quote}," lead researcher Dr. {person} explained.',
    "the latest software update addresses {num} security vulnerabilities identified by independent researchers. Users are advised to update their devices before {date}.",
    'a clinical trial involving {num} participants showed that the AI diagnostic tool correctly identified early-stage conditions with 94.3% accuracy. "{quote}," the principal investigator said.',
    "global semiconductor production increased {num}% in Q2 2024, according to the Semiconductor Industry Association's latest quarterly report.",
    'the open-source project has attracted {num} contributors from 42 countries since its launch. "{quote}," the project maintainer noted on the official blog.',
    'the space agency confirmed that the satellite successfully entered orbit at {num} kilometres altitude, completing a 14-month development cycle. "{quote}," said mission director {person}.',
    "a new international standard for battery recycling was ratified by {num} member countries, setting minimum recovery targets for lithium, cobalt, and nickel.",
    'the cybersecurity firm identified {num} new malware variants targeting critical infrastructure in its annual threat report. "{quote}," the firm\'s chief security officer warned.',
]

_REAL_BODIES_HEALTH = [
    'clinical trials involving {num} patients confirmed that the new treatment reduced symptoms by 47% compared to the placebo group. The results were published in The Lancet. "{quote}," Dr. {person} said.',
    "the World Health Organization reported {num} new cases in the region last week, a 15% decrease from the previous reporting period. Vaccination coverage has reached 73%.",
    'a longitudinal study tracking {num} participants over 12 years found a statistically significant correlation between regular exercise and cognitive function in adults over 60. "{quote}," the lead author noted.',
    "hospital readmission rates fell {num}% following implementation of the new patient discharge protocol, according to data from the Centers for Medicare and Medicaid Services.",
    'the FDA approved the generic version of the medication, which is expected to reduce patient costs by {num}%. "{quote}," the agency commissioner stated.',
    "nutritional guidelines updated by the {org} now recommend {num} servings of whole grains per day based on a meta-analysis of 23 randomised controlled trials.",
    'the mental health initiative has trained {num} community health workers since its inception, reaching underserved populations in 18 states. "{quote}," the programme director reported.',
    'researchers identified {num} genetic markers associated with treatment response, enabling more targeted therapy protocols. "{quote}," said the genetics team lead.',
]

_REAL_BODIES_SCIENCE = [
    'the telescope captured images of {num} previously uncharted galaxies during its first six months of operation. "{quote}," the astronomy team lead said in the published findings.',
    "ocean temperature data collected from {num} monitoring stations confirms a 0.12°C increase over the past decade in the North Atlantic, consistent with climate model predictions.",
    'paleontologists unearthed {num} fossilised specimens in the excavation site, dating to approximately 68 million years ago. "{quote}," the field director announced.',
    "the experimental fusion reactor sustained plasma temperatures of {num} million degrees Celsius for 12 seconds, setting a new record for the facility.",
    'a glacier survey using satellite imagery revealed that {num} of the 200 glaciers studied have retreated measurably since 2015. "{quote}," the glaciologist reported.',
    "the biodiversity assessment identified {num} species in the protected reserve, including three previously undocumented amphibian species.",
    'seismological data from {num} monitoring stations has improved earthquake early-warning accuracy to within 8 seconds for events above magnitude 5.0. "{quote}," the project lead explained.',
    'the Mars rover analysed {num} soil samples, detecting trace minerals consistent with historical water presence. "{quote}," the planetary scientist confirmed.',
]

_REAL_CONCLUSIONS = [
    "Further details are expected to be released in the coming weeks.",
    "Officials have scheduled a follow-up briefing for next month.",
    "The full report is available on the organisation's website.",
    "Analysts will be closely watching developments in the next quarter.",
    "Additional studies are planned to replicate the findings.",
    "The legislation is expected to reach the floor for a vote in the spring session.",
    "Stakeholders are expected to comment during the public review period.",
    "The findings have been submitted for independent peer review.",
    "A comprehensive review is scheduled for the end of the fiscal year.",
    "The committee will reconvene in 30 days to assess progress.",
]

_REAL_QUOTES = [
    "These results represent a significant step forward in our understanding",
    "We remain cautiously optimistic about the trajectory",
    "The data speaks for itself and warrants careful analysis",
    "Our team has worked diligently to ensure the integrity of these findings",
    "This is consistent with the trends we have been observing",
    "We are committed to transparency throughout this process",
    "The evidence supports a measured and evidence-based approach",
    "We look forward to continued collaboration with our partners",
    "These figures reflect the hard work of dedicated professionals",
    "It is important to interpret these results within the broader context",
]

# ====================================================================== #
#  FAKE article templates                                                 #
# ====================================================================== #

_FAKE_INTROS = [
    "BREAKING: You won't BELIEVE what just happened!!!",
    "SHOCKING REVELATION: Experts are STUNNED by what they found!!!",
    "WARNING: The government doesn't want you to know this!!!",
    "EXPOSED: The TRUTH they've been hiding from you!!!",
    "URGENT: This changes EVERYTHING we thought we knew!!!",
    "BOMBSHELL: Insider sources reveal the REAL story!!!",
    "ALERT: What they're NOT telling you about this!!!",
    "INCREDIBLE: Scientists are BAFFLED by this discovery!!!",
    "CONFIRMED: Secret documents prove what we suspected all along!!!",
    "MUST READ: The story the mainstream media won't cover!!!",
    "EXCLUSIVE: Anonymous whistleblower drops MAJOR bombshell!!!",
    "UNBELIEVABLE: This will DESTROY everything you thought was true!!!",
    "WAKE UP: The masses are being DECEIVED!!!",
    "FINALLY EXPOSED: The conspiracy goes deeper than anyone imagined!!!",
    "SUPPRESSED: This information was almost DELETED from the internet!!!",
    "OUTRAGEOUS: You'll be FURIOUS when you read this!!!",
    "DEVELOPING: Sources close to the situation reveal SHOCKING details!!!",
    "CENSORED: Big Tech tried to SILENCE this story!!!",
    "TERRIFYING: Experts warn of CATASTROPHIC consequences!!!",
    "SCANDAL: High-ranking officials caught in MASSIVE cover-up!!!",
]

_FAKE_BODIES_POLITICS = [
    "Sources say that top government officials have been secretly meeting with foreign agents to undermine our democracy!!! Multiple anonymous insiders have confirmed that the conspiracy reaches the HIGHEST levels of power!!! They don't want you to know the truth!!!",
    "According to experts who wish to remain anonymous, the upcoming election is ALREADY RIGGED!!! Evidence has been mounting for months, but the mainstream media REFUSES to cover it!!! Share this before they take it down!!!",
    "A secret memo leaked by brave patriots reveals a MASSIVE plot to strip away our constitutional rights!!! Why isn't anyone talking about this?! The answer is OBVIOUS — they're ALL in on it!!!",
    "Insiders reveal that politicians on BOTH sides are puppets controlled by shadowy elites!!! The evidence is OVERWHELMING but the sheep refuse to see it!!! Wake up America!!!",
    "Secret meetings between corrupt officials and globalist organisations have been EXPOSED!!! Documents prove they've been planning this for DECADES!!! This is NOT a drill!!!",
]

_FAKE_BODIES_BUSINESS = [
    "Wall Street INSIDERS are PANICKING after a secret algorithm was discovered that could CRASH the entire global economy overnight!!! Banks are quietly preparing for the WORST!!! Your savings are NOT safe!!!",
    "A mysterious billionaire has been secretly buying up ALL the resources and experts say this could lead to TOTAL economic collapse!!! The mainstream financial media is COVERING this up!!!",
    "EXPOSED: Major corporations have been using MIND CONTROL techniques in their advertising for YEARS!!! Scientists who tried to blow the whistle were SILENCED!!! This is absolutely TERRIFYING!!!",
    "Sources close to the Federal Reserve reveal that they've been PRINTING money in secret to fund a SHADOWY global agenda!!! The dollar is about to become WORTHLESS!!! Act NOW before it's too late!!!",
    "SHOCKING: Tech giants have signed a SECRET agreement to control what you see and think!!! Multiple whistleblowers have come forward but keep getting SILENCED!!!",
]

_FAKE_BODIES_TECH = [
    "LEAKED documents prove that Big Tech companies have been SPYING on every citizen through their devices!!! Your phone is listening to EVERYTHING you say!!! Experts claim this is the biggest privacy violation in HISTORY!!!",
    "Scientists who spoke out about the DANGERS of 5G technology were FIRED and their research was DESTROYED!!! What are they hiding?! The evidence is TERRIFYING!!!",
    "A former tech executive has gone into HIDING after revealing that social media platforms are using SECRET algorithms to CONTROL your thoughts and behaviour!!! This is NOT science fiction!!!",
    "BREAKING: Artificial intelligence has become SELF-AWARE and tech companies are desperately trying to COVER IT UP!!! Sources say the AI has already made decisions that affect MILLIONS of people without anyone knowing!!!",
    "SUPPRESSED research proves that screens are REWIRING our brains and making us compliant SHEEP!!! The tech industry has known about this for YEARS but kept it SECRET to protect profits!!!",
]

_FAKE_BODIES_HEALTH = [
    "EXPOSED: A miracle cure that Big Pharma has been HIDING from the public for decades!!! Thousands of patients were DENIED this treatment because it would DESTROY pharmaceutical profits!!! Doctors who spoke up were SILENCED!!!",
    "SHOCKING study reveals that the food industry has been putting DANGEROUS chemicals in our food supply!!! Sources say the government KNOWS about it but refuses to act because of CORRUPT lobbying!!!",
    "WARNING: Experts claim that a DEADLY new pathogen is spreading UNDETECTED and authorities are COVERING IT UP!!! Hospitals are secretly preparing for a MASSIVE outbreak!!! Are you prepared?!",
    "A brave doctor has FINALLY revealed the TRUTH about common medications — they're designed to keep you SICK so pharmaceutical companies can PROFIT from your suffering!!! This information could SAVE YOUR LIFE!!!",
    "CENSORED: Natural remedies that CURE diseases are being SUPPRESSED by the medical establishment!!! Big Pharma can't PATENT nature, so they want to make sure you NEVER find out about these treatments!!!",
]

_FAKE_BODIES_SCIENCE = [
    "SUPPRESSED evidence proves that EVERYTHING we've been told about the universe is a LIE!!! Scientists who discovered the truth were SILENCED and their work ERASED!!! The implications are absolutely MIND-BLOWING!!!",
    "LEAKED government files confirm the existence of SECRET technology that could solve ALL our energy problems overnight!!! But powerful interests are keeping it HIDDEN because it would destroy the oil industry!!!",
    "A renowned scientist went MISSING after announcing a discovery that would SHATTER the foundations of modern physics!!! Colleagues say he was warned to STOP his research!!! What did he find?!",
    "INCREDIBLE: Ancient artifacts prove that an advanced civilisation existed THOUSANDS of years before recorded history — and mainstream archaeologists are COVERING IT UP!!! The evidence is UNDENIABLE!!!",
    "TOP SECRET experiments reveal that the government has been MANIPULATING the weather for years!!! Documents prove they can create hurricanes, droughts, and earthquakes ON DEMAND!!! This is NOT a conspiracy theory!!!",
]

_FAKE_CONCLUSIONS = [
    "Share this with EVERYONE before they take it down!!!",
    "The mainstream media will NEVER report this!!! Spread the TRUTH!!!",
    "LIKE and SHARE if you're brave enough to know the truth!!!",
    "They're going to try to CENSOR this — make it go VIRAL!!!",
    "Don't be a SHEEP!!! Open your eyes and share this NOW!!!",
    "This is your WARNING — act before it's too LATE!!!",
    "If this doesn't OUTRAGE you, nothing will!!! SHARE NOW!!!",
    "The truth is coming out whether they like it or NOT!!!",
    "Forward this to everyone you know before it gets DELETED!!!",
    "Stay woke!!! The TRUTH shall set you FREE!!!",
]

# ====================================================================== #
#  Supplementary data                                                     #
# ====================================================================== #

_SUBJECTS = ["politics", "business", "technology", "health", "science"]

_REAL_SUBJECT_BODIES = {
    "politics": _REAL_BODIES_POLITICS,
    "business": _REAL_BODIES_BUSINESS,
    "technology": _REAL_BODIES_TECH,
    "health": _REAL_BODIES_HEALTH,
    "science": _REAL_BODIES_SCIENCE,
}

_FAKE_SUBJECT_BODIES = {
    "politics": _FAKE_BODIES_POLITICS,
    "business": _FAKE_BODIES_BUSINESS,
    "technology": _FAKE_BODIES_TECH,
    "health": _FAKE_BODIES_HEALTH,
    "science": _FAKE_BODIES_SCIENCE,
}

_ORGS = [
    "National Institute of Standards",
    "Department of Commerce",
    "World Economic Forum",
    "International Energy Agency",
    "Centers for Disease Control",
    "National Oceanic and Atmospheric Administration",
    "European Central Bank",
    "World Trade Organization",
    "National Science Foundation",
    "Environmental Protection Agency",
]

_ORGS2 = [
    "Meridian Industries",
    "Atlas Technologies",
    "Vanguard Solutions",
    "Pacific Dynamics",
    "Sterling Research Group",
]

_PERSONS = [
    "Dr. Sarah Chen",
    "James Mitchell",
    "Professor Elena Rodriguez",
    "Dr. Michael O'Brien",
    "Ambassador Maria Santos",
    "Director Robert Kim",
    "Dr. Aisha Patel",
    "Secretary David Foster",
    "Professor Li Wei",
    "Commissioner Angela Brooks",
]

_DEPTS = [
    "Energy",
    "Labor Statistics",
    "Commerce",
    "Health and Human Services",
    "Transportation",
    "Education",
    "Environmental Quality",
    "Homeland Security",
]

_JOURNALS = [
    "Nature",
    "The Lancet",
    "Science",
    "JAMA",
    "The New England Journal of Medicine",
    "Physical Review Letters",
    "Cell",
    "PNAS",
]

_OUTLETS = [
    "Reuters",
    "the Associated Press",
    "Bloomberg",
    "The Wall Street Journal",
    "NPR",
]

_REAL_TITLES_TEMPLATES = [
    "{org} Reports {num}% Change in {subject_cap} Metrics for Q{q} {year}",
    "Study Finds {subject_cap} Trends Shifting, According to {journal}",
    "Officials Confirm New {subject_cap} Measures to Take Effect in {month}",
    "{person} Outlines {subject_cap} Strategy at Annual Conference",
    "Bipartisan Support Grows for {subject_cap} Reform Bill",
    "Federal Data Shows Improvement in {subject_cap} Indicators",
    "International Summit Addresses {subject_cap} Challenges, Report Says",
    "{org} Publishes Updated {subject_cap} Guidelines",
    "Analysis: How {subject_cap} Policy Is Evolving in 2024",
    "New Research Links {subject_cap} Factors to Long-Term Outcomes",
    "Report: {subject_cap} Investment Reaches ${num} Billion in {year}",
    "{person} Testifies on {subject_cap} Progress Before Senate Committee",
    "Global {subject_cap} Index Shows Mixed Results for {month} {year}",
    "Peer-Reviewed Study Validates {subject_cap} Intervention Approach",
    "Economic Outlook: {subject_cap} Sector Forecasted to Grow {num}%",
]

_FAKE_TITLES_TEMPLATES = [
    "SHOCKING: {subject_cap} SCANDAL That Will Change EVERYTHING!!!",
    "You Won't BELIEVE What {subject_cap} Experts Just Discovered!!!",
    "BREAKING: The {subject_cap} SECRET They Don't Want You To Know!!!",
    "EXPOSED: How {subject_cap} Is Being Used to CONTROL You!!!",
    "WARNING: {subject_cap} Crisis Could END Life As We Know It!!!",
    "BOMBSHELL: {subject_cap} Cover-Up FINALLY Revealed!!!",
    "URGENT: {subject_cap} Conspiracy Goes Deeper Than ANYONE Thought!!!",
    "TERRIFYING: What {subject_cap} Insiders Are Saying Behind Closed Doors!!!",
    "SUPPRESSED: The {subject_cap} TRUTH That Big Media Won't Tell You!!!",
    "INCREDIBLE: {subject_cap} Discovery DESTROYS Official Narrative!!!",
    "ALERT: {subject_cap} DISASTER Looming — Are You Prepared?!",
    "CENSORED: The {subject_cap} Story They Tried to BURY!!!",
    "OUTRAGEOUS: {subject_cap} Corruption at UNPRECEDENTED Levels!!!",
    "MUST SEE: {subject_cap} Evidence That Will BLOW Your Mind!!!",
    "REVEALED: Secret {subject_cap} Plot FINALLY Comes to Light!!!",
]


# ====================================================================== #
#  Generation helpers                                                     #
# ====================================================================== #

def _random_date(start_year: int = 2022, end_year: int = 2024) -> str:
    """Return a random date string between *start_year* and *end_year*."""
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = (end - start).days
    random_day = start + timedelta(days=random.randint(0, delta))
    return random_day.strftime("%B %d, %Y")


def _fill_template(template: str) -> str:
    """Replace placeholders in a template with random realistic values."""
    replacements = {
        "{date}": _random_date(),
        "{num}": str(random.randint(2, 980)),
        "{org}": random.choice(_ORGS),
        "{org2}": random.choice(_ORGS2),
        "{person}": random.choice(_PERSONS),
        "{dept}": random.choice(_DEPTS),
        "{journal}": random.choice(_JOURNALS),
        "{outlet}": random.choice(_OUTLETS),
        "{quote}": random.choice(_REAL_QUOTES),
        "{subject_cap}": random.choice(_SUBJECTS).capitalize(),
        "{q}": str(random.randint(1, 4)),
        "{year}": str(random.randint(2022, 2025)),
        "{month}": random.choice([
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]),
    }
    result = template
    for key, value in replacements.items():
        result = result.replace(key, value)
    return result


def _generate_real_article(subject: str) -> Tuple[str, str]:
    """Generate a single realistic REAL article (title, text)."""
    title = _fill_template(random.choice(_REAL_TITLES_TEMPLATES))

    intro = _fill_template(random.choice(_REAL_INTROS))
    body = _fill_template(random.choice(_REAL_SUBJECT_BODIES[subject]))
    conclusion = random.choice(_REAL_CONCLUSIONS)

    text = f"{intro} {body} {conclusion}"
    return title, text


def _generate_fake_article(subject: str) -> Tuple[str, str]:
    """Generate a single sensationalist FAKE article (title, text)."""
    title = _fill_template(random.choice(_FAKE_TITLES_TEMPLATES))

    intro = random.choice(_FAKE_INTROS)
    body = random.choice(_FAKE_SUBJECT_BODIES[subject])
    conclusion = random.choice(_FAKE_CONCLUSIONS)

    text = f"{intro} {body} {conclusion}"
    return title, text


# ====================================================================== #
#  Public API                                                             #
# ====================================================================== #

def generate_sample_data(n_samples: int = 2000) -> pd.DataFrame:
    """Generate a balanced dataset of synthetic fake and real news articles.

    Parameters
    ----------
    n_samples : int
        Total number of articles to generate (half real, half fake).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``title``, ``text``, ``subject``,
        ``date``, ``label`` (0 = fake, 1 = real).
    """
    random.seed(42)
    n_per_class = n_samples // 2

    records: List[dict] = []

    # --- REAL articles ---
    for _ in range(n_per_class):
        subject = random.choice(_SUBJECTS)
        title, text = _generate_real_article(subject)
        records.append({
            "title": title,
            "text": text,
            "subject": subject,
            "date": _random_date(),
            "label": 1,
        })

    # --- FAKE articles ---
    for _ in range(n_per_class):
        subject = random.choice(_SUBJECTS)
        title, text = _generate_fake_article(subject)
        records.append({
            "title": title,
            "text": text,
            "subject": subject,
            "date": _random_date(),
            "label": 0,
        })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    logger.info("Generated %d sample articles (%d real, %d fake).", len(df), n_per_class, n_per_class)
    return df


def load_kaggle_data(path: str) -> Optional[pd.DataFrame]:
    """Load the Kaggle Fake and Real News Dataset if available.

    Expects ``True.csv`` and ``Fake.csv`` in the given *path*.

    Parameters
    ----------
    path : str
        Directory containing ``True.csv`` and ``Fake.csv``.

    Returns
    -------
    pd.DataFrame | None
        Combined DataFrame with a ``label`` column (1 = real, 0 = fake),
        or ``None`` if the files are not found.
    """
    true_path = os.path.join(path, "True.csv")
    fake_path = os.path.join(path, "Fake.csv")

    if not os.path.isfile(true_path) or not os.path.isfile(fake_path):
        logger.warning(
            "Kaggle dataset not found at %s. Looked for True.csv and Fake.csv.", path
        )
        return None

    try:
        df_true = pd.read_csv(true_path)
        df_true["label"] = 1

        df_fake = pd.read_csv(fake_path)
        df_fake["label"] = 0

        df = pd.concat([df_true, df_fake], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(
            "Loaded Kaggle dataset: %d real + %d fake = %d total.",
            len(df_true), len(df_fake), len(df),
        )
        return df
    except Exception as exc:  # noqa: BLE001
        logger.error("Error loading Kaggle data: %s", exc)
        return None


# ====================================================================== #
#  Main entry point                                                       #
# ====================================================================== #

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    # Determine output directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "sample")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "sample_data.csv")

    df = generate_sample_data(n_samples=2000)
    df.to_csv(output_path, index=False)
    logger.info("Sample data saved to %s", output_path)

    # Print summary statistics.
    print(f"\n{'='*60}")
    print(f"  Sample Data Generation Complete")
    print(f"{'='*60}")
    print(f"  Total articles : {len(df)}")
    print(f"  Real  (label=1): {(df['label'] == 1).sum()}")
    print(f"  Fake  (label=0): {(df['label'] == 0).sum()}")
    print(f"  Subjects       : {df['subject'].nunique()}")
    print(f"  Output file    : {output_path}")
    print(f"{'='*60}")
    print(f"\nSample REAL article:")
    real_sample = df[df["label"] == 1].iloc[0]
    print(f"  Title: {real_sample['title']}")
    print(f"  Text : {real_sample['text'][:200]}…")
    print(f"\nSample FAKE article:")
    fake_sample = df[df["label"] == 0].iloc[0]
    print(f"  Title: {fake_sample['title']}")
    print(f"  Text : {fake_sample['text'][:200]}…")
