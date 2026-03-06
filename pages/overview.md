<div style="font-family: 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: auto;">

<h1 style="text-align: center; color: #1a73e8;">🏥 Diabetic Retinopathy Detection Challenge</h1>

<p style="text-align: center; font-size: 1.15em; color: #555;">
  Classify retinal fundus images into <strong>5 severity levels</strong> of diabetic retinopathy using machine learning.
</p>

<hr style="border: 1px solid #e0e0e0;">

<h2 style="color: #1a73e8;">🎯 Objective</h2>

<p>
  <strong>Diabetic retinopathy (DR)</strong> is a diabetes complication that damages the blood vessels of the retina.
  It is a leading cause of blindness worldwide, affecting over <strong>100 million people</strong>.
  Early detection through automated screening of retinal images can drastically improve patient outcomes.
</p>

<p>
  Your goal is to build a model that takes a <strong>retinal fundus photograph</strong> as input and predicts
  which <strong>severity level</strong> of DR it corresponds to.
</p>

<h2 style="color: #1a73e8;">📊 Severity Levels</h2>

<table style="width: 100%; border-collapse: collapse; margin: 15px 0;">
  <thead>
    <tr style="background: #1a73e8; color: white;">
      <th style="padding: 10px; text-align: center;">Label</th>
      <th style="padding: 10px; text-align: left;">Severity</th>
      <th style="padding: 10px; text-align: left;">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background: #f8f9fa;">
      <td style="padding: 8px; text-align: center; font-weight: bold;">0</td>
      <td style="padding: 8px;">No DR</td>
      <td style="padding: 8px;">Healthy retina, no signs of diabetic retinopathy</td>
    </tr>
    <tr>
      <td style="padding: 8px; text-align: center; font-weight: bold;">1</td>
      <td style="padding: 8px;">Mild</td>
      <td style="padding: 8px;">Microaneurysms only</td>
    </tr>
    <tr style="background: #f8f9fa;">
      <td style="padding: 8px; text-align: center; font-weight: bold;">2</td>
      <td style="padding: 8px;">Moderate</td>
      <td style="padding: 8px;">More than just microaneurysms, but less than severe</td>
    </tr>
    <tr>
      <td style="padding: 8px; text-align: center; font-weight: bold;">3</td>
      <td style="padding: 8px;">Severe</td>
      <td style="padding: 8px;">Extensive intraretinal hemorrhages, venous beading</td>
    </tr>
    <tr style="background: #f8f9fa;">
      <td style="padding: 8px; text-align: center; font-weight: bold;">4</td>
      <td style="padding: 8px;">Proliferative DR</td>
      <td style="padding: 8px;">Neovascularization and/or vitreous hemorrhage</td>
    </tr>
  </tbody>
</table>

<h2 style="color: #1a73e8;">📁 Dataset</h2>

<p>
  The dataset consists of <strong>510 retinal fundus images</strong> from the
  <a href="https://universe.roboflow.com/officeworkspace/diabetic-retinopathy-dataset" target="_blank">IDRiD (Indian Diabetic Retinopathy Image Dataset)</a>,
  available under the <strong>CC BY 4.0</strong> license.
</p>

<ul>
  <li><strong>Training set</strong>: 407 images with labels</li>
  <li><strong>Public test set</strong> (Development Phase): ~50 images</li>
  <li><strong>Private test set</strong> (Final Phase): ~50 images</li>
</ul>

<p>
  Labels are provided as <strong>one-hot encoded vectors</strong> of length 5 (columns <code>0</code> to <code>4</code>).
  Each image belongs to exactly one severity class.
</p>

<h2 style="color: #1a73e8;">📐 Evaluation Metrics</h2>

<p>Submissions are evaluated on the following metrics:</p>

<table style="width: 100%; border-collapse: collapse; margin: 15px 0;">
  <thead>
    <tr style="background: #34a853; color: white;">
      <th style="padding: 10px; text-align: left;">Metric</th>
      <th style="padding: 10px; text-align: left;">Description</th>
      <th style="padding: 10px; text-align: center;">Direction</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background: #f8f9fa;">
      <td style="padding: 8px;"><strong>F1 Macro</strong></td>
      <td style="padding: 8px;">Average F1 score across all classes (treats each class equally)</td>
      <td style="padding: 8px; text-align: center;">↑ Higher is better</td>
    </tr>
    <tr>
      <td style="padding: 8px;"><strong>F1 Micro</strong></td>
      <td style="padding: 8px;">Global F1 computed from total TP, FP, FN (favors majority classes)</td>
      <td style="padding: 8px; text-align: center;">↑ Higher is better</td>
    </tr>
    <tr style="background: #f8f9fa;">
      <td style="padding: 8px;"><strong>Hamming Loss</strong></td>
      <td style="padding: 8px;">Fraction of incorrectly predicted labels</td>
      <td style="padding: 8px; text-align: center;">↓ Lower is better</td>
    </tr>
    <tr>
      <td style="padding: 8px;"><strong>Runtime (s)</strong></td>
      <td style="padding: 8px;">Total execution time (training + inference)</td>
      <td style="padding: 8px; text-align: center;">↓ Lower is better</td>
    </tr>
  </tbody>
</table>

<p>
  Predictions are <strong>thresholded at 0.5</strong> to produce binary labels before computing F1 and Hamming Loss.
</p>

<h2 style="color: #1a73e8;">🚀 How to Participate</h2>

<ol>
  <li>Download the <strong>starting kit notebook</strong> (<code>starting_kit.ipynb</code>) from the GitHub repository.</li>
  <li>Explore the data and understand the evaluation metrics.</li>
  <li>Implement your model in a <code>submission.py</code> file following the required interface:
    <ul>
      <li>A <code>get_model()</code> function that returns a model object</li>
      <li>The model must have <code>fit(X_train, y_train, train_img_dir)</code> and <code>predict(X_test, test_img_dir)</code> methods</li>
      <li><code>predict</code> should return a numpy array of shape <code>(n_samples, 5)</code> with values in [0, 1]</li>
    </ul>
  </li>
  <li>Zip your <code>submission.py</code> and upload it to CodaBench.</li>
</ol>

<h2 style="color: #1a73e8;">⚠️ Rules</h2>

<div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px; margin: 15px 0; border-radius: 4px;">
  <strong>Development Phase</strong>: Up to <strong>100 submissions</strong> total (5 per day).<br>
  <strong>Final Phase</strong>: Up to <strong>3 submissions</strong> total (1 per day).
</div>

<h2 style="color: #1a73e8;">📚 References</h2>

<ul>
  <li><a href="https://universe.roboflow.com/officeworkspace/diabetic-retinopathy-dataset" target="_blank">IDRiD Dataset (Roboflow)</a></li>
  <li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html" target="_blank">Scikit-learn F1 Score Documentation</a></li>
</ul>

</div>
