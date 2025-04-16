import { StagehandPage } from "../StagehandPage";
import { AgentProvider } from "../agent/AgentProvider";
import { StagehandAgent } from "../agent/StagehandAgent";
import { AgentClient } from "../agent/AgentClient";
import { LogLine } from "../../types/log";
import { Page } from "playwright";
import {
  AgentExecuteOptions,
  AgentAction,
  AgentResult,
  AgentHandlerOptions,
  ActionExecutionResult,
} from "@/types/agent";

export class StagehandAgentHandler {
  private stagehandPage: StagehandPage;
  private agent: StagehandAgent;
  private provider: AgentProvider;
  private logger: (message: LogLine) => void;
  private agentClient: AgentClient;
  private options: AgentHandlerOptions;

  constructor(
    stagehandPage: StagehandPage,
    logger: (message: LogLine) => void,
    options: AgentHandlerOptions,
  ) {
    this.stagehandPage = stagehandPage;
    this.logger = logger;
    this.options = options;

    // Initialize the provider
    this.provider = new AgentProvider(logger);

    // Create client first
    const client = this.provider.getClient(
      options.modelName,
      options.clientOptions || {},
      options.userProvidedInstructions,
    );

    // Store the client
    this.agentClient = client;

    // Set up common functionality for any client type
    this.setupAgentClient();

    // Create agent with the client
    this.agent = new StagehandAgent(client, logger);
  }

  private setupAgentClient(): void {
    // Set up screenshot provider for any client type
    this.agentClient.setScreenshotProvider(async () => {
      const screenshot = await this.stagehandPage.page.screenshot({
        fullPage: false,
      });
      // Convert to base64
      return screenshot.toString("base64");
    });

    // Set up action handler for any client type
    this.agentClient.setActionHandler(async (action) => {
      // Default delay between actions (1 second if not specified)
      const defaultDelay = 1000;
      // Use specified delay or default
      const waitBetweenActions =
        (this.options.clientOptions?.waitBetweenActions as number) ||
        defaultDelay;

      try {
        // Try to inject cursor before each action
        try {
          await this.injectCursor();
        } catch {
          // Ignore cursor injection failures
        }

        // Add a small delay before the action for better visibility
        await new Promise((resolve) => setTimeout(resolve, 500));

        // Execute the action
        await this.executeAction(action);

        // Add a delay after the action for better visibility
        await new Promise((resolve) => setTimeout(resolve, waitBetweenActions));

        // After executing an action, take a screenshot
        try {
          await this.captureAndSendScreenshot();
        } catch (error) {
          const errorMessage =
            error instanceof Error ? error.message : String(error);
          this.logger({
            category: "agent",
            message: `Warning: Failed to take screenshot after action: ${errorMessage}. Continuing execution.`,
            level: 1,
          });
          // Continue execution even if screenshot fails
        }
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : String(error);
        this.logger({
          category: "agent",
          message: `Error executing action ${action.type}: ${errorMessage}`,
          level: 0,
        });
        throw error; // Re-throw the error to be handled by the caller
      }
    });

    // Update viewport and URL for any client type
    this.updateClientViewport();
    this.updateClientUrl();
  }

  /**
   * Execute a task with the agent
   */
  async execute(
    optionsOrInstruction: AgentExecuteOptions | string,
  ): Promise<AgentResult> {
    const options =
      typeof optionsOrInstruction === "string"
        ? { instruction: optionsOrInstruction }
        : optionsOrInstruction;

    //Redirect to Google if the URL is empty or about:blank
    const currentUrl = this.stagehandPage.page.url();
    if (!currentUrl || currentUrl === "about:blank") {
      this.logger({
        category: "agent",
        message: `Page URL is empty or about:blank. Redirecting to www.google.com...`,
        level: 0,
      });
      await this.stagehandPage.page.goto("https://www.google.com");
    }

    this.logger({
      category: "agent",
      message: `Executing agent task: ${options.instruction}`,
      level: 1,
    });

    // Inject cursor for visual feedback
    try {
      await this.injectCursor();
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      this.logger({
        category: "agent",
        message: `Warning: Failed to inject cursor: ${errorMessage}. Continuing with execution.`,
        level: 1,
      });
      // Continue execution even if cursor injection fails
    }

    // Take initial screenshot if needed
    if (options.autoScreenshot !== false) {
      try {
        await this.captureAndSendScreenshot();
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : String(error);
        this.logger({
          category: "agent",
          message: `Warning: Failed to take initial screenshot: ${errorMessage}. Continuing with execution.`,
          level: 1,
        });
        // Continue execution even if screenshot fails
      }
    }

    // Execute the task
    const result = await this.agent.execute(optionsOrInstruction);

    // The actions are now executed during the agent's execution flow
    // We don't need to execute them again here

    return result;
  }

  /**
   * Execute a single action on the page
   */
  private async executeAction(
    action: AgentAction,
  ): Promise<ActionExecutionResult> {
    try {
      switch (action.type) {
        case "click": {
          const { x, y, button = "left" } = action;
          // Update cursor position first
          await this.updateCursorPosition(x as number, y as number);
          // Animate the click
          await this.animateClick(x as number, y as number);
          // Small delay to see the animation
          await new Promise((resolve) => setTimeout(resolve, 300));
          // Perform the actual click
          await this.stagehandPage.page.mouse.click(x as number, y as number, {
            button: button as "left" | "right",
          });
          const newOpenedTab = await Promise.race([
            new Promise<Page | null>((resolve) => {
              this.stagehandPage.context.once("page", (page) => resolve(page));
              setTimeout(() => resolve(null), 1500);
            }),
          ]);
          if (newOpenedTab) {
            this.logger({
              category: "action",
              message: `New page detected (new tab) with URL. Opening on current page...`,
              level: 1,
              auxiliary: {
                url: {
                  value: newOpenedTab.url(),
                  type: "string",
                },
              },
            });
            await newOpenedTab.close();
            await this.stagehandPage.page.goto(newOpenedTab.url());
            await this.stagehandPage.page.waitForURL(newOpenedTab.url());
          }
          return { success: true };
        }

        case "double_click": {
          const { x, y } = action;
          // Update cursor position first
          await this.updateCursorPosition(x as number, y as number);
          // Animate the click
          await this.animateClick(x as number, y as number);
          // Small delay to see the animation
          await new Promise((resolve) => setTimeout(resolve, 200));
          // Animate the second click
          await this.animateClick(x as number, y as number);
          // Small delay to see the animation
          await new Promise((resolve) => setTimeout(resolve, 200));
          // Perform the actual double click
          await this.stagehandPage.page.mouse.dblclick(
            x as number,
            y as number,
          );
          return { success: true };
        }

        // Handle the case for "doubleClick" as well for backward compatibility
        case "doubleClick": {
          const { x, y } = action;
          // Update cursor position first
          await this.updateCursorPosition(x as number, y as number);
          // Animate the click
          await this.animateClick(x as number, y as number);
          // Small delay to see the animation
          await new Promise((resolve) => setTimeout(resolve, 200));
          // Animate the second click
          await this.animateClick(x as number, y as number);
          // Small delay to see the animation
          await new Promise((resolve) => setTimeout(resolve, 200));
          // Perform the actual double click
          await this.stagehandPage.page.mouse.dblclick(
            x as number,
            y as number,
          );
          return { success: true };
        }

        case "type": {
          const { text } = action;
          await this.stagehandPage.page.keyboard.type(text as string);
          return { success: true };
        }

        case "keypress": {
          const { keys } = action;
          if (Array.isArray(keys)) {
            for (const key of keys) {
              // Handle special keys
              if (key.includes("ENTER")) {
                await this.stagehandPage.page.keyboard.press("Enter");
              } else if (key.includes("SPACE")) {
                await this.stagehandPage.page.keyboard.press(" ");
              } else if (key.includes("TAB")) {
                await this.stagehandPage.page.keyboard.press("Tab");
              } else if (key.includes("ESCAPE") || key.includes("ESC")) {
                await this.stagehandPage.page.keyboard.press("Escape");
              } else if (key.includes("BACKSPACE")) {
                await this.stagehandPage.page.keyboard.press("Backspace");
              } else if (key.includes("DELETE")) {
                await this.stagehandPage.page.keyboard.press("Delete");
              } else if (key.includes("ARROW_UP")) {
                await this.stagehandPage.page.keyboard.press("ArrowUp");
              } else if (key.includes("ARROW_DOWN")) {
                await this.stagehandPage.page.keyboard.press("ArrowDown");
              } else if (key.includes("ARROW_LEFT")) {
                await this.stagehandPage.page.keyboard.press("ArrowLeft");
              } else if (key.includes("ARROW_RIGHT")) {
                await this.stagehandPage.page.keyboard.press("ArrowRight");
              } else {
                // For other keys, use the existing conversion
                const playwrightKey = this.convertKeyName(key);
                await this.stagehandPage.page.keyboard.press(playwrightKey);
              }
            }
          }
          return { success: true };
        }

        case "scroll": {
          const { x, y, scroll_x = 0, scroll_y = 0 } = action;
          // First move to the position
          await this.stagehandPage.page.mouse.move(x as number, y as number);
          // Then scroll
          await this.stagehandPage.page.evaluate(
            ({ scrollX, scrollY }) => window.scrollBy(scrollX, scrollY),
            { scrollX: scroll_x as number, scrollY: scroll_y as number },
          );
          return { success: true };
        }

        case "drag": {
          const { path } = action;
          if (Array.isArray(path) && path.length >= 2) {
            const start = path[0];

            // Update cursor position for start
            await this.updateCursorPosition(start.x, start.y);
            await this.stagehandPage.page.mouse.move(start.x, start.y);
            await this.stagehandPage.page.mouse.down();

            // Update cursor position for each point in the path
            for (let i = 1; i < path.length; i++) {
              await this.updateCursorPosition(path[i].x, path[i].y);
              await this.stagehandPage.page.mouse.move(path[i].x, path[i].y);
            }

            await this.stagehandPage.page.mouse.up();
          }
          return { success: true };
        }

        case "move": {
          const { x, y } = action;
          // Update cursor position first
          await this.updateCursorPosition(x as number, y as number);
          await this.stagehandPage.page.mouse.move(x as number, y as number);
          return { success: true };
        }

        case "wait": {
          await new Promise((resolve) => setTimeout(resolve, 1000));
          return { success: true };
        }

        case "screenshot": {
          // Screenshot is handled automatically by the agent client
          // after each action, so we don't need to do anything here
          return { success: true };
        }

        case "function": {
          const { name, arguments: args = {} } = action;

          if (
            name === "goto" &&
            typeof args === "object" &&
            args !== null &&
            "url" in args
          ) {
            await this.stagehandPage.page.goto(args.url as string);
            this.updateClientUrl();
            return { success: true };
          } else if (name === "back") {
            await this.stagehandPage.page.goBack();
            this.updateClientUrl();
            return { success: true };
          } else if (name === "forward") {
            await this.stagehandPage.page.goForward();
            this.updateClientUrl();
            return { success: true };
          } else if (name === "reload") {
            await this.stagehandPage.page.reload();
            this.updateClientUrl();
            return { success: true };
          }

          return {
            success: false,
            error: `Unsupported function: ${name}`,
          };
        }

        case "key": {
          // Handle the 'key' action type from Anthropic
          const { text } = action;
          if (text === "Return" || text === "Enter") {
            await this.stagehandPage.page.keyboard.press("Enter");
          } else if (text === "Tab") {
            await this.stagehandPage.page.keyboard.press("Tab");
          } else if (text === "Escape" || text === "Esc") {
            await this.stagehandPage.page.keyboard.press("Escape");
          } else if (text === "Backspace") {
            await this.stagehandPage.page.keyboard.press("Backspace");
          } else {
            // For other keys, try to press directly
            await this.stagehandPage.page.keyboard.press(text as string);
          }
          return { success: true };
        }

        default:
          return {
            success: false,
            error: `Unsupported action type: ${action.type}`,
          };
      }
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);

      this.logger({
        category: "agent",
        message: `Error executing action ${action.type}: ${errorMessage}`,
        level: 0,
      });

      return {
        success: false,
        error: errorMessage,
      };
    }
  }

  private updateClientViewport(): void {
    const viewportSize = this.stagehandPage.page.viewportSize();
    if (viewportSize) {
      this.agentClient.setViewport(viewportSize.width, viewportSize.height);
    }
  }

  private updateClientUrl(): void {
    const url = this.stagehandPage.page.url();
    this.agentClient.setCurrentUrl(url);
  }

  getAgent(): StagehandAgent {
    return this.agent;
  }

  getClient(): AgentClient {
    return this.agentClient;
  }

  async captureAndSendScreenshot(): Promise<unknown> {
    this.logger({
      category: "agent",
      message: "Taking screenshot and sending to agent",
      level: 1,
    });

    try {
      // Take screenshot of the current page
      const screenshot = await this.stagehandPage.page.screenshot({
        type: "png",
        fullPage: false,
      });

      // Convert to base64
      const base64Image = screenshot.toString("base64");

      // Just use the captureScreenshot method on the agent client
      return await this.agentClient.captureScreenshot({
        base64Image,
        currentUrl: this.stagehandPage.page.url(),
      });
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      this.logger({
        category: "agent",
        message: `Error capturing screenshot: ${errorMessage}`,
        level: 0,
      });
      return null;
    }
  }

  /**
   * Inject a cursor element into the page for visual feedback
   */
  private async injectCursor(): Promise<void> {
    try {
      // Define constants for cursor and highlight element IDs
      const CURSOR_ID = "stagehand-cursor";
      const HIGHLIGHT_ID = "stagehand-highlight";

      // Check if cursor already exists
      const cursorExists = await this.stagehandPage.page.evaluate(
        (id: string) => {
          return !!document.getElementById(id);
        },
        CURSOR_ID,
      );

      if (cursorExists) {
        return;
      }

      // Inject cursor and highlight elements
      await this.stagehandPage.page.evaluate(`
        (function(cursorId, highlightId) {
          // Create cursor element
          const cursor = document.createElement('div');
          cursor.id = cursorId;
          
          // Use the provided SVG for a custom cursor
          cursor.innerHTML = \`
          <svg xmlns="http://www.w3.org/2000/svg" version="1.1" viewBox="0 0 28 28" width="28" height="28">
            <polygon fill="#000000" points="9.2,7.3 9.2,18.5 12.2,15.6 12.6,15.5 17.4,15.5"/>
            <rect x="12.5" y="13.6" transform="matrix(0.9221 -0.3871 0.3871 0.9221 -5.7605 6.5909)" width="2" height="8" fill="#000000"/>
          </svg>
          \`;
          
          // Style the cursor
          cursor.style.position = 'absolute';
          cursor.style.top = '0';
          cursor.style.left = '0';
          cursor.style.width = '28px';
          cursor.style.height = '28px';
          cursor.style.pointerEvents = 'none';
          cursor.style.zIndex = '9999999';
          cursor.style.transform = 'translate(-4px, -4px)'; // Adjust to align the pointer tip
          
          // Create highlight element for click animation
          const highlight = document.createElement('div');
          highlight.id = highlightId;
          highlight.style.position = 'absolute';
          highlight.style.width = '20px';
          highlight.style.height = '20px';
          highlight.style.borderRadius = '50%';
          highlight.style.backgroundColor = 'rgba(66, 134, 244, 0)';
          highlight.style.transform = 'translate(-50%, -50%) scale(0)';
          highlight.style.pointerEvents = 'none';
          highlight.style.zIndex = '9999998';
          highlight.style.transition = 'transform 0.3s ease-out, opacity 0.3s ease-out';
          highlight.style.opacity = '0';
          
          // Add elements to the document
          document.body.appendChild(cursor);
          document.body.appendChild(highlight);
          
          // Add a function to update cursor position
          window.__updateCursorPosition = function(x, y) {
            if (cursor) {
              cursor.style.transform = \`translate(\${x - 4}px, \${y - 4}px)\`;
            }
          };
          
          // Add a function to animate click
          window.__animateClick = function(x, y) {
            if (highlight) {
              highlight.style.left = \`\${x}px\`;
              highlight.style.top = \`\${y}px\`;
              highlight.style.transform = 'translate(-50%, -50%) scale(1)';
              highlight.style.opacity = '1';
              
              setTimeout(() => {
                highlight.style.transform = 'translate(-50%, -50%) scale(0)';
                highlight.style.opacity = '0';
              }, 300);
            }
          };
        })('${CURSOR_ID}', '${HIGHLIGHT_ID}');
      `);

      this.logger({
        category: "agent",
        message: "Cursor injected for visual feedback",
        level: 1,
      });
    } catch (error) {
      this.logger({
        category: "agent",
        message: `Failed to inject cursor: ${error}`,
        level: 0,
      });
    }
  }

  /**
   * Update the cursor position on the page
   */
  private async updateCursorPosition(x: number, y: number): Promise<void> {
    try {
      await this.stagehandPage.page.evaluate(
        ({ x, y }) => {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          if ((window as any).__updateCursorPosition) {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            (window as any).__updateCursorPosition(x, y);
          }
        },
        { x, y },
      );
    } catch {
      // Silently fail if cursor update fails
      // This is not critical functionality
    }
  }

  /**
   * Animate a click at the given position
   */
  private async animateClick(x: number, y: number): Promise<void> {
    try {
      await this.stagehandPage.page.evaluate(
        ({ x, y }) => {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          if ((window as any).__animateClick) {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            (window as any).__animateClick(x, y);
          }
        },
        { x, y },
      );
    } catch {
      // Silently fail if animation fails
      // This is not critical functionality
    }
  }

  private convertKeyName(key: string): string {
    // Map of CUA key names to Playwright key names
    const keyMap: Record<string, string> = {
      ENTER: "Enter",
      ESCAPE: "Escape",
      BACKSPACE: "Backspace",
      TAB: "Tab",
      SPACE: " ",
      ARROWUP: "ArrowUp",
      ARROWDOWN: "ArrowDown",
      ARROWLEFT: "ArrowLeft",
      ARROWRIGHT: "ArrowRight",
      UP: "ArrowUp",
      DOWN: "ArrowDown",
      LEFT: "ArrowLeft",
      RIGHT: "ArrowRight",
      SHIFT: "Shift",
      CONTROL: "Control",
      ALT: "Alt",
      META: "Meta",
      COMMAND: "Meta",
      CMD: "Meta",
      CTRL: "Control",
      DELETE: "Delete",
      HOME: "Home",
      END: "End",
      PAGEUP: "PageUp",
      PAGEDOWN: "PageDown",
    };

    // Convert to uppercase for case-insensitive matching
    const upperKey = key.toUpperCase();

    // Return the mapped key or the original key if not found
    return keyMap[upperKey] || key;
  }
}
