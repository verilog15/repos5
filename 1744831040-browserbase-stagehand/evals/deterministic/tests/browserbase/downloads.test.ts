import { test, expect } from "@playwright/test";
import AdmZip from "adm-zip";
import StagehandConfig from "@/evals/deterministic/stagehand.config";
import { Stagehand } from "@/dist";
import Browserbase from "@browserbasehq/sdk";

const downloadRe = /sandstorm-(\d{13})+\.mp3/;

test("Downloads", async () => {
  const stagehand = new Stagehand(StagehandConfig);
  await stagehand.init();
  const page = stagehand.page;
  const context = stagehand.context;

  const client = await context.newCDPSession(page);
  await client.send("Browser.setDownloadBehavior", {
    behavior: "allow",
    // `downloadPath` gets appended to the browser's default download directory.
    // set to "downloads", it ends up being "/app/apps/browser/downloads/<file>".
    downloadPath: "downloads",
    eventsEnabled: true,
  });

  await page.goto("https://browser-tests-alpha.vercel.app/api/download-test");

  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.locator("#download").click(),
  ]);

  const downloadError = await download.failure();

  await stagehand.close();

  if (downloadError !== null) {
    throw new Error(
      `Download for session ${stagehand.browserbaseSessionID} failed: ${downloadError}`,
    );
  }

  expect(async () => {
    const bb = new Browserbase();
    const zipBuffer = await bb.sessions.downloads.list(
      stagehand.browserbaseSessionID,
    );
    if (!zipBuffer) {
      throw new Error(
        `Download buffer is empty for session ${stagehand.browserbaseSessionID}`,
      );
    }

    const zip = new AdmZip(Buffer.from(await zipBuffer.arrayBuffer()));
    const zipEntries = zip.getEntries();
    const mp3Entry = zipEntries.find((entry) =>
      downloadRe.test(entry.entryName),
    );

    if (!mp3Entry) {
      throw new Error(
        `Session ${stagehand.browserbaseSessionID} is missing a file matching "${downloadRe.toString()}" in its zip entries: ${JSON.stringify(zipEntries.map((entry) => entry.entryName))}`,
      );
    }

    const expectedFileSize = 6137541;
    expect(mp3Entry.header.size).toBe(expectedFileSize);
  }).toPass({
    timeout: 30_000,
  });
});
