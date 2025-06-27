// server.ts
import express from "express";
import { createProxyMiddleware } from "http-proxy-middleware";
import path from "path";

const app = express();
const PORT = 3000;
app.use(express.static(path.join(__dirname, '../public')));


app.get(/.*/, (_req, res) => {
  res.sendFile(path.join(__dirname, '../public/index.html'));
});

app.listen(PORT, () =>
  console.log(`Express proxy running at http://localhost:${PORT}`)
);
