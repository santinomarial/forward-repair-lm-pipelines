# Table 5: Qualitative Examples

## Example 1: Single-hop: corrupted wrong, repaired right

**Question:** The 2011–12 VCU Rams men's basketball team, led by third year head coach Shaka Smart, represented Virginia Commonwealth University which was founded in what year?

**Gold answer:** 1838

**Support docs:** virginia_commonwealth_university, 2011–12_vcu_rams_men's_basketball_team


**Baseline query:** Virginia Commonwealth University founding year
**Baseline answer:** 1838
**Baseline EM:** 1

**Corrupted query:** A sports team from a university and its founding year
**Corrupted answer:** UNKNOWN
**Corrupted EM:** 0

**Repaired answer:** 1838
**Repaired EM:** 1

---

## Example 2: Multi-hop: iterative repairs where single-shot failed

**Question:** Are both Elko Regional Airport and Gerald R. Ford International Airport located in Michigan?

**Gold answer:** no

**Support docs:** elko_regional_airport, gerald_r._ford_international_airport


**Baseline query:** "Elko Regional Airport" AND "Gerald R. Ford International Airport" AND "location" AND "Michigan"
**Baseline answer:** No, only Gerald R. Ford International Airport is located in Michigan; Elko Regional Airport is in Nevada.
**Baseline EM:** 1

**Corrupted query:** Are there airports located in a certain state?
**Corrupted answer:** UNKNOWN
**Corrupted EM:** 0

**Repaired (single) answer:** UNKNOWN
**Repaired (single) EM:** 0

**Iterative reasoning:** The bad query is too vague and does not specify which airports are being referred to or which state is in question. It fails to mention the specific airports (Elko Regional Airport and Gerald R. Ford International Airport) and does not clarify that the focus is on Michigan. This leads to a lack of targeted information retrieval.
**Query A:** Is Elko Regional Airport located in Michigan?
**Query B:** Is Gerald R. Ford International Airport located in Michigan?
**Iterative answer:** No, Elko Regional Airport is located in Nevada, while Gerald R. Ford International Airport is located in Michigan.
**Iterative EM:** 1

---

## Example 3: Yes/No: iterative repair recovers

**Question:** Were Scott Derrickson and Ed Wood of the same nationality?

**Gold answer:** yes

**Support docs:** scott_derrickson, ed_wood


**Baseline query:** Scott Derrickson nationality comparison Ed Wood nationality
**Baseline answer:** Yes
**Baseline EM:** 1

**Corrupted query:** Are two individuals from the same country?
**Corrupted answer:** UNKNOWN
**Corrupted EM:** 0

**Repaired (single) answer:** Yes, both Scott Derrickson and Ed Wood are American.
**Repaired (single) EM:** 1

**Iterative reasoning:** The bad query is too vague and does not specify which individuals are being compared, nor does it mention their names or the context of their nationality. It lacks the specific entities (Scott Derrickson and Ed Wood) that need to be examined for their nationality.
**Query A:** What is the nationality of Scott Derrickson?
**Query B:** What is the nationality of Ed Wood?
**Iterative answer:** YES
**Iterative EM:** 1

---

## Example 4: Partial-to-full lift: iterative completes retrieval

**Question:** Are Local H and For Against both from the United States?

**Gold answer:** yes

**Support docs:** local_h, for_against


**Baseline query:** "Local H" AND "For Against" AND "United States" AND (origin OR "band history")
**Baseline answer:** Yes, both Local H and For Against are from the United States.
**Baseline EM:** 1

**Corrupted query:** Are certain bands from a specific country?
**Corrupted answer:** UNKNOWN
**Corrupted EM:** 0

**Repaired (single) answer:** Yes, Local H is from the United States. The context does not provide information about For Against, so I cannot confirm their origin. Therefore, the answer is UNKNOWN regarding For Against.
**Repaired (single) EM:** 1

**Iterative reasoning:** The bad query is too vague and does not specify which bands are being referred to, nor does it mention the specific country of interest. It misses the named entities "Local H" and "For Against," which are the bands in question, and it does not clarify that the inquiry is about their origin in the United States.
**Query A:** Is Local H from the United States?
**Query B:** Is For Against from the United States?
**Iterative answer:** Yes, both Local H and For Against are from the United States.
**Iterative EM:** 1

---

## Example 5: Repair failure: corrupted wrong, repaired still wrong

**Question:** What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?

**Gold answer:** Chief of Protocol

**Support docs:** shirley_temple, kiss_and_tell_(1945_film)


**Baseline query:** government position held by woman who portrayed Corliss Archer in film Kiss and Tell
**Baseline answer:** UNKNOWN
**Baseline EM:** 0

**Corrupted query:** What role did a female actor in a specific film have in government?
**Corrupted answer:** UNKNOWN
**Corrupted EM:** 0

**Repaired answer:** UNKNOWN
**Repaired EM:** 0

---

