import { z } from "zod";
import { LogLine } from "../types/log";
import { ChatMessage, LLMClient } from "./llm/LLMClient";
import {
  buildExtractSystemPrompt,
  buildExtractUserPrompt,
  buildMetadataPrompt,
  buildMetadataSystemPrompt,
  buildObserveSystemPrompt,
  buildObserveUserMessage,
  buildRefineSystemPrompt,
  buildRefineUserPrompt,
} from "./prompt";
import {
  appendSummary,
  writeTimestampedTxtFile,
} from "@/lib/inferenceLogUtils";

/**
 * Replaces <|VARIABLE|> placeholders in a text with user-provided values.
 */
export function fillInVariables(
  text: string,
  variables: Record<string, string>,
) {
  let processedText = text;
  Object.entries(variables).forEach(([key, value]) => {
    const placeholder = `<|${key.toUpperCase()}|>`;
    processedText = processedText.replace(placeholder, value);
  });
  return processedText;
}

/** Simple usage shape if your LLM returns usage tokens. */
interface LLMUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

/**
 * For calls that use a schema: the LLMClient may return { data: T; usage?: LLMUsage }
 */
export interface LLMParsedResponse<T> {
  data: T;
  usage?: LLMUsage;
}

export async function extract({
  instruction,
  previouslyExtractedContent,
  domElements,
  schema,
  llmClient,
  chunksSeen,
  chunksTotal,
  requestId,
  logger,
  isUsingTextExtract,
  userProvidedInstructions,
  logInferenceToFile = false,
}: {
  instruction: string;
  previouslyExtractedContent: object;
  domElements: string;
  schema: z.ZodObject<z.ZodRawShape>;
  llmClient: LLMClient;
  chunksSeen: number;
  chunksTotal: number;
  requestId: string;
  isUsingTextExtract?: boolean;
  userProvidedInstructions?: string;
  logger: (message: LogLine) => void;
  logInferenceToFile?: boolean;
}) {
  const metadataSchema = z.object({
    progress: z
      .string()
      .describe(
        "progress of what has been extracted so far, as concise as possible",
      ),
    completed: z
      .boolean()
      .describe(
        "true if the goal is now accomplished. Use this conservatively, only when sure that the goal has been completed.",
      ),
  });

  type ExtractionResponse = z.infer<typeof schema>;
  type MetadataResponse = z.infer<typeof metadataSchema>;

  const isUsingAnthropic = llmClient.type === "anthropic";

  const extractCallMessages: ChatMessage[] = [
    buildExtractSystemPrompt(
      isUsingAnthropic,
      isUsingTextExtract,
      userProvidedInstructions,
    ),
    buildExtractUserPrompt(instruction, domElements, isUsingAnthropic),
  ];

  let extractCallFile = "";
  let extractCallTimestamp = "";
  if (logInferenceToFile) {
    const { fileName, timestamp } = writeTimestampedTxtFile(
      "extract_summary",
      "extract_call",
      {
        requestId,
        modelCall: "extract",
        messages: extractCallMessages,
      },
    );
    extractCallFile = fileName;
    extractCallTimestamp = timestamp;
  }

  const extractStartTime = Date.now();
  const extractionResponse =
    await llmClient.createChatCompletion<ExtractionResponse>({
      options: {
        messages: extractCallMessages,
        response_model: {
          schema,
          name: "Extraction",
        },
        temperature: 0.1,
        top_p: 1,
        frequency_penalty: 0,
        presence_penalty: 0,
        requestId,
      },
      logger,
    });
  const extractEndTime = Date.now();

  const { data: extractedData, usage: extractUsage } =
    extractionResponse as LLMParsedResponse<ExtractionResponse>;

  let extractResponseFile = "";
  if (logInferenceToFile) {
    const { fileName } = writeTimestampedTxtFile(
      "extract_summary",
      "extract_response",
      {
        requestId,
        modelResponse: "extract",
        rawResponse: extractedData,
      },
    );
    extractResponseFile = fileName;

    appendSummary("extract", {
      extract_inference_type: "extract",
      timestamp: extractCallTimestamp,
      LLM_input_file: extractCallFile,
      LLM_output_file: extractResponseFile,
      prompt_tokens: extractUsage?.prompt_tokens ?? 0,
      completion_tokens: extractUsage?.completion_tokens ?? 0,
      inference_time_ms: extractEndTime - extractStartTime,
    });
  }

  const refineCallMessages: ChatMessage[] = [
    buildRefineSystemPrompt(),
    buildRefineUserPrompt(
      instruction,
      previouslyExtractedContent,
      extractedData,
    ),
  ];

  let refineCallFile = "";
  let refineCallTimestamp = "";
  if (logInferenceToFile) {
    const { fileName, timestamp } = writeTimestampedTxtFile(
      "extract_summary",
      "refine_call",
      {
        requestId,
        modelCall: "refine",
        messages: refineCallMessages,
      },
    );
    refineCallFile = fileName;
    refineCallTimestamp = timestamp;
  }

  const refineStartTime = Date.now();
  const refinedResponse =
    await llmClient.createChatCompletion<ExtractionResponse>({
      options: {
        messages: refineCallMessages,
        response_model: {
          schema,
          name: "RefinedExtraction",
        },
        temperature: 0.1,
        top_p: 1,
        frequency_penalty: 0,
        presence_penalty: 0,
        requestId,
      },
      logger,
    });
  const refineEndTime = Date.now();

  const { data: refinedResponseData, usage: refinedResponseUsage } =
    refinedResponse as LLMParsedResponse<ExtractionResponse>;

  let refineResponseFile = "";
  if (logInferenceToFile) {
    const { fileName } = writeTimestampedTxtFile(
      "extract_summary",
      "refine_response",
      {
        requestId,
        modelResponse: "refine",
        rawResponse: refinedResponseData,
      },
    );
    refineResponseFile = fileName;

    appendSummary("extract", {
      extract_inference_type: "refine",
      timestamp: refineCallTimestamp,
      LLM_input_file: refineCallFile,
      LLM_output_file: refineResponseFile,
      prompt_tokens: refinedResponseUsage?.prompt_tokens ?? 0,
      completion_tokens: refinedResponseUsage?.completion_tokens ?? 0,
      inference_time_ms: refineEndTime - refineStartTime,
    });
  }

  const metadataCallMessages: ChatMessage[] = [
    buildMetadataSystemPrompt(),
    buildMetadataPrompt(
      instruction,
      refinedResponseData,
      chunksSeen,
      chunksTotal,
    ),
  ];

  let metadataCallFile = "";
  let metadataCallTimestamp = "";
  if (logInferenceToFile) {
    const { fileName, timestamp } = writeTimestampedTxtFile(
      "extract_summary",
      "metadata_call",
      {
        requestId,
        modelCall: "metadata",
        messages: metadataCallMessages,
      },
    );
    metadataCallFile = fileName;
    metadataCallTimestamp = timestamp;
  }

  const metadataStartTime = Date.now();
  const metadataResponse =
    await llmClient.createChatCompletion<MetadataResponse>({
      options: {
        messages: metadataCallMessages,
        response_model: {
          name: "Metadata",
          schema: metadataSchema,
        },
        temperature: 0.1,
        top_p: 1,
        frequency_penalty: 0,
        presence_penalty: 0,
        requestId,
      },
      logger,
    });
  const metadataEndTime = Date.now();

  const {
    data: {
      completed: metadataResponseCompleted,
      progress: metadataResponseProgress,
    },
    usage: metadataResponseUsage,
  } = metadataResponse as LLMParsedResponse<MetadataResponse>;

  let metadataResponseFile = "";
  if (logInferenceToFile) {
    const { fileName } = writeTimestampedTxtFile(
      "extract_summary",
      "metadata_response",
      {
        requestId,
        modelResponse: "metadata",
        completed: metadataResponseCompleted,
        progress: metadataResponseProgress,
      },
    );
    metadataResponseFile = fileName;

    appendSummary("extract", {
      extract_inference_type: "metadata",
      timestamp: metadataCallTimestamp,
      LLM_input_file: metadataCallFile,
      LLM_output_file: metadataResponseFile,
      prompt_tokens: metadataResponseUsage?.prompt_tokens ?? 0,
      completion_tokens: metadataResponseUsage?.completion_tokens ?? 0,
      inference_time_ms: metadataEndTime - metadataStartTime,
    });
  }

  const totalPromptTokens =
    (extractUsage?.prompt_tokens ?? 0) +
    (refinedResponseUsage?.prompt_tokens ?? 0) +
    (metadataResponseUsage?.prompt_tokens ?? 0);

  const totalCompletionTokens =
    (extractUsage?.completion_tokens ?? 0) +
    (refinedResponseUsage?.completion_tokens ?? 0) +
    (metadataResponseUsage?.completion_tokens ?? 0);

  const totalInferenceTimeMs =
    extractEndTime -
    extractStartTime +
    (refineEndTime - refineStartTime) +
    (metadataEndTime - metadataStartTime);

  return {
    ...refinedResponseData,
    metadata: {
      completed: metadataResponseCompleted,
      progress: metadataResponseProgress,
    },
    prompt_tokens: totalPromptTokens,
    completion_tokens: totalCompletionTokens,
    inference_time_ms: totalInferenceTimeMs,
  };
}

export async function observe({
  instruction,
  domElements,
  llmClient,
  requestId,
  isUsingAccessibilityTree,
  userProvidedInstructions,
  logger,
  returnAction = false,
  logInferenceToFile = false,
  fromAct,
}: {
  instruction: string;
  domElements: string;
  llmClient: LLMClient;
  requestId: string;
  userProvidedInstructions?: string;
  logger: (message: LogLine) => void;
  isUsingAccessibilityTree?: boolean;
  returnAction?: boolean;
  logInferenceToFile?: boolean;
  fromAct?: boolean;
}) {
  const observeSchema = z.object({
    elements: z
      .array(
        z.object({
          elementId: z.number().describe("the number of the element"),
          description: z
            .string()
            .describe(
              isUsingAccessibilityTree
                ? "a description of the accessible element and its purpose"
                : "a description of the element and what it is relevant for",
            ),
          ...(returnAction
            ? {
                method: z
                  .string()
                  .describe(
                    "the candidate method/action to interact with the element. Select one of the available Playwright interaction methods.",
                  ),
                arguments: z.array(
                  z
                    .string()
                    .describe(
                      "the arguments to pass to the method. For example, for a click, the arguments are empty, but for a fill, the arguments are the value to fill in.",
                    ),
                ),
              }
            : {}),
        }),
      )
      .describe(
        isUsingAccessibilityTree
          ? "an array of accessible elements that match the instruction"
          : "an array of elements that match the instruction",
      ),
  });

  type ObserveResponse = z.infer<typeof observeSchema>;

  const messages: ChatMessage[] = [
    buildObserveSystemPrompt(
      userProvidedInstructions,
      isUsingAccessibilityTree,
    ),
    buildObserveUserMessage(instruction, domElements, isUsingAccessibilityTree),
  ];

  const filePrefix = fromAct ? "act" : "observe";
  let callTimestamp = "";
  let callFile = "";
  if (logInferenceToFile) {
    const { fileName, timestamp } = writeTimestampedTxtFile(
      `${filePrefix}_summary`,
      `${filePrefix}_call`,
      {
        requestId,
        modelCall: filePrefix,
        messages,
      },
    );
    callFile = fileName;
    callTimestamp = timestamp;
  }

  const start = Date.now();
  const rawResponse = await llmClient.createChatCompletion<ObserveResponse>({
    options: {
      messages,
      response_model: {
        schema: observeSchema,
        name: "Observation",
      },
      temperature: 0.1,
      top_p: 1,
      frequency_penalty: 0,
      presence_penalty: 0,
      requestId,
    },
    logger,
  });
  const end = Date.now();
  const usageTimeMs = end - start;

  const { data: observeData, usage: observeUsage } =
    rawResponse as LLMParsedResponse<ObserveResponse>;
  const promptTokens = observeUsage?.prompt_tokens ?? 0;
  const completionTokens = observeUsage?.completion_tokens ?? 0;

  let responseFile = "";
  if (logInferenceToFile) {
    const { fileName: responseFileName } = writeTimestampedTxtFile(
      `${filePrefix}_summary`,
      `${filePrefix}_response`,
      {
        requestId,
        modelResponse: filePrefix,
        rawResponse: observeData,
      },
    );
    responseFile = responseFileName;

    appendSummary(filePrefix, {
      [`${filePrefix}_inference_type`]: filePrefix,
      timestamp: callTimestamp,
      LLM_input_file: callFile,
      LLM_output_file: responseFile,
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      inference_time_ms: usageTimeMs,
    });
  }

  const parsedElements =
    observeData.elements?.map((el) => {
      const base = {
        elementId: Number(el.elementId),
        description: String(el.description),
      };
      if (returnAction) {
        return {
          ...base,
          method: String(el.method),
          arguments: el.arguments,
        };
      }
      return base;
    }) ?? [];

  return {
    elements: parsedElements,
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    inference_time_ms: usageTimeMs,
  };
}
