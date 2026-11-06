// Create new notebooks only. Never overwrite a learner's work or source CSVs.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import reviewQueue from './roblox-review-queue-spec.mjs';
import playtimeTail from './roblox-playtime-tail-spec.mjs';
import requestRanking from './roblox-request-ranking-spec.mjs';
import highEngagement from './roblox-high-engagement-spec.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const all = [reviewQueue, playtimeTail, requestRanking, highEngagement];
const selected = process.argv.find(value => value.startsWith('--only='))?.slice(7);
if (selected && !all.some(mock => mock.slug === selected)) throw new Error('Unknown mock selection.');

const imports = `from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn import metrics

SEED = 42`;

export function notebook(mock, solved) {
  if (mock.steps.length !== 7) throw new Error(`${mock.slug} must have seven steps.`);
  const cleanTitle = step => step.title.replace(/^\d+\.\s*/, '');
  const cleanTime = step => step.time.replace(/\s+min$/, '');
  const highlights = Array.isArray(mock.highlights)
    ? mock.highlights.map(value => `- ${value}`).join('\n') : mock.highlights;
  const cells = [];
  function add(kind, suffix, source, tag) {
    const cell = {cell_type: kind, id: `${mock.slug}-${suffix}`,
      metadata: tag ? {tags: [tag]} : {}, source: source.trim() + '\n'};
    if (kind === 'code') Object.assign(cell, {execution_count: null, outputs: []});
    cells.push(cell);
  }
  add('markdown', 'intro', `# Data Analysis & Modeling: ${mock.title}

**Original mock inspired by collected interview topics, not a reported Roblox coding question.** All data are synthetic. Format: **5 minutes introduction + 35 minutes coding + 5 minutes Q&A**. Explain your assumptions and choices while coding.

## Interviewer prompt

${mock.prompt}

${highlights}

| Coding time | Focus |
| --- | --- |
${mock.steps.map(step => `| ${cleanTime(step)} min | ${cleanTitle(step)} |`).join('\n')}

Suggested checkpoints only; they are not official grading criteria. The two problems are data analysis (Steps 1–3) and modeling (Steps 4–7).

${solved ? '**Reference version:** complete code, checks, saved outputs and expandable follow-up answers. Try the practice notebook first.' : '**Practice version:** write your own imports, then complete each exercise and run its check cell. The questions are part of the mock; explain your answer before moving on. Run All intentionally stops at unfinished exercises.'}

## Data contract

${mock.contract}`);
  add('markdown', 'setup-note', `## Setup

Use the workspace root as the working directory and select **.venv-codesignal/bin/python**. Inputs are in **${mock.dataDir}/**. No downloads, data generation, GPU or new dependencies are needed.

${solved ? 'Imports and the data path are supplied below.' : 'Write the imports you need, define **SEED = 42**, and set **DATA_DIR** to the supplied directory. Include setup in your timed attempt. Restart the kernel first so earlier notebooks do not hide missing imports.'}

Do not edit the source CSVs. These exercises deliberately reuse existing files to focus on new decisions rather than a new dataset generator.`);
  add('code', 'setup', solved
    ? `${imports}\nDATA_DIR = Path(${JSON.stringify(mock.dataDir)})\nassert DATA_DIR.is_dir(), "Open the notebook from the workspace root."`
    : '# Write your own imports, SEED and DATA_DIR.\nraise NotImplementedError("Complete setup before loading the CSVs.")',
  solved ? 'provided' : 'setup-exercise');
  mock.steps.forEach((step, index) => {
    const number = index + 1;
    add('markdown', `step-${number}-prompt`, `## Problem ${index < 3 ? 1 : 2} · Step ${number}: ${cleanTitle(step)} (${cleanTime(step)} min)

${step.task}

**Say before coding:** "${step.say}"

**Possible interviewer questions**

${step.questions.map(([question], i) => `${i + 1}. ${question}`).join('\n')}`);
    add('code', `step-${number}-code`, solved ? step.code
      : `# Implement the task above.\nraise NotImplementedError("Complete step ${number}, then run its checks.")`,
    solved ? 'solution' : 'exercise');
    add('code', `step-${number}-check`, step.check, 'check');
    if (solved) add('markdown', `step-${number}-answers`, `<details>
<summary>Follow-up answers</summary>

${step.questions.map(([question, answer]) => `**${question}**\n\n${answer}`).join('\n\n')}

</details>`);
  });
  if (mock.extension) {
    add('markdown', 'extension', `## Optional follow-ups — outside the 35-minute core\n\n${mock.extension.prompt}`);
    if (solved && mock.extension.code) add('code', 'extension-code', mock.extension.code, 'optional');
  }
  add('markdown', 'sources', `## Scope and sources

The user-supplied **Data Analysis & Modeling Coding Interview** guide supplies the format: data loading, analysis, feature engineering, a baseline model, evaluation and live explanation in a CodeSignal notebook. It does not specify these exact tasks or APIs.

The local interview collection separates reported **Question/Asked** topics from our explanatory **Answer** material. This mock adapts the topics for practice; it is not evidence that Roblox used this coding problem. Synthetic rates and thresholds are not real product statistics or policy.

${mock.sourceNotes}`);
  return {cells, metadata: {kernelspec: {display_name: 'Python 3', language: 'python', name: 'python3'},
    language_info: {name: 'python', version: '3.9'}}, nbformat: 4, nbformat_minor: 5};
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
const targets = all.filter(mock => !selected || mock.slug === selected).flatMap(mock =>
  [false, true].map(solved => ({mock, solved,
    file: path.join(root, `codesignal_roblox_${mock.slug}_${solved ? 'solution' : 'practice'}.ipynb`)})));
for (const target of targets) {
  if (fs.existsSync(target.file)) throw new Error(`Refusing to overwrite ${path.basename(target.file)}.`);
}
// Prepare every notebook before writing, so invalid specs cannot leave a partial set.
const prepared = targets.map(target => ({...target, content: JSON.stringify(notebook(target.mock, target.solved), null, 1) + '\n'}));
for (const target of prepared) {
  fs.writeFileSync(target.file, target.content, {flag: 'wx'});
  console.log(path.basename(target.file));
}
}
