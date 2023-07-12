import { OpenAI } from "langchain/llms/openai";
import { loadSummarizationChain } from "langchain/chains";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import * as fs from "fs";



const run = async () => {
    // In this example, we use a `MapReduceDocumentsChain` specifically prompted to summarize a set of documents.
  let openai_api_key = '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';
  const text = fs.readFileSync("state_of_the_union.txt", "utf8");
  const model = new OpenAI({ temperature: 0, openAIApiKey: openai_api_key });
  const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 100, chunkOverlap: 0 });
  const docs = await textSplitter.createDocuments([text]);

  // This convenience function creates a document chain prompted to summarize a set of documents.
  const chain = loadSummarizationChain(model, { type: "stuff" });
  const res = await chain.call({
    input_documents: docs,
  });
  console.log({ res });
  /*
  {
    res: {
      text: ' President Biden is taking action to protect Americans from the COVID-19 pandemic and Russian aggression, providing economic relief, investing in infrastructure, creating jobs, and fighting inflation.
      He is also proposing measures to reduce the cost of prescription drugs, protect voting rights, and reform the immigration system. The speaker is advocating for increased economic security, police reform, and the Equality Act, as well as providing support for veterans and military families.
      The US is making progress in the fight against COVID-19, and the speaker is encouraging Americans to come together and work towards a brighter future.'
    }
  }
  */
};

run();