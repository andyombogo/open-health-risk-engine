const payload = {
  age: 35,
  sex_female: 0,
  poverty_ratio: 2.5,
  met_min_week: 300,
  sleep_hours: 7.0,
  sleep_trouble: 0,
  bmi: 24.0,
  drinks_per_week: 3,
  education: 4,
  race_eth: 3,
};

async function scoreRisk() {
  const response = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": "replace-with-a-strong-local-key",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Prediction failed with status ${response.status}`);
  }

  const result = await response.json();
  console.log("Predicted risk", result.risk_score, result.risk_label);
  return result;
}

scoreRisk().catch((error) => {
  console.error(error);
});
