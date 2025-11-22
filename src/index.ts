import cors, { CorsOptions } from "cors";
import dotenv from "dotenv";
import express, { Request, Response } from "express";
import mongoose, { Schema } from "mongoose";
import { GoogleGenAI } from "@google/genai";

dotenv.config();

const PORT = Number(process.env.PORT ?? 5000);
const MONGO_URL = process.env.MONGO_URL ?? "mongodb://localhost:27017/labelcraft";
const GEMINI_MODEL = process.env.GEMINI_MODEL ?? "gemini-2.5-flash";

const rawOrigins = process.env.APP_ORIGIN;
const allowedOrigins = rawOrigins
  ?.split(",")
  .map(origin => origin.trim())
  .filter(Boolean);

const corsOptions: CorsOptions = allowedOrigins && allowedOrigins.length
  ? {
      origin: (origin, callback) => {
        if (!origin || allowedOrigins.includes("*") || allowedOrigins.includes(origin)) {
          callback(null, true);
        } else {
          callback(new Error("Not allowed by CORS"));
        }
      }
    }
  : { origin: true };

const app = express();
app.use(cors(corsOptions));
app.use(express.json({ limit: process.env.BODY_LIMIT ?? "10mb" }));
app.use(express.urlencoded({ extended: true, limit: process.env.BODY_LIMIT ?? "10mb" }));

const labelSchema = new Schema(
  {
    context: { type: String, required: true },
    productType: { type: String, required: true },
    text: { type: String, required: true },
    createdAt: { type: Date, default: Date.now }
  },
  { versionKey: false }
);

const LabelModel = mongoose.model("Label", labelSchema);

const folderSchema = new Schema(
  {
    name: { type: String, required: true },
    parentId: { type: String, default: null }
  },
  { versionKey: false, timestamps: true }
);

const canvasItemSchema = new Schema(
  {
    id: { type: String, required: true },
    type: { type: String, enum: ["TEXT", "QR", "IMAGE"], required: true },
    content: { type: String, default: "" },
    x: { type: Number, required: true },
    y: { type: Number, required: true },
    width: { type: Number, required: true },
    height: { type: Number, required: true },
    fontSize: Number,
    fontFamily: String,
    fontWeight: String,
    fontStyle: String,
    textDecoration: String,
    color: String,
    textAlign: String
  },
  { _id: false }
);

const templateFieldSchema = new Schema(
  {
    name: { type: String, required: true },
    label: { type: String, required: true },
    targetItemId: { type: String, required: true }
  },
  { _id: false }
);

const templateSchema = new Schema(
  {
    name: { type: String, required: true },
    folderId: { type: String, default: null },
    width: { type: Number, required: true },
    height: { type: Number, required: true },
    items: { type: [canvasItemSchema], default: [] },
    fields: { type: [templateFieldSchema], default: [] },
    role: { type: String, enum: ["draft", "filled"], default: "draft" }
  },
  { versionKey: false, timestamps: true }
);

const FolderModel = mongoose.model("Folder", folderSchema);
const TemplateModel = mongoose.model("Template", templateSchema);

type CanvasItemPayload = {
  id: string;
  type: "TEXT" | "QR" | "IMAGE";
  content: string;
  x: number;
  y: number;
  width: number;
  height: number;
  fontSize?: number;
  fontFamily?: string;
  fontWeight?: string;
  fontStyle?: string;
  textDecoration?: string;
  color?: string;
  textAlign?: string;
};

type TemplatePayload = {
  name?: string;
  folderId?: string | null;
  width?: number;
  height?: number;
  items?: CanvasItemPayload[];
};

type TemplateCreatePayload = Required<Pick<TemplatePayload, "name" | "width" | "height" | "items">> & {
  folderId: string | null;
};

type TemplateFieldPayload = {
  name: string;
  label: string;
  targetItemId: string;
};

type MergeRow = Record<string, string>;

type FolderResponse = {
  id: string;
  name: string;
  parentId: string | null;
};

type TemplateResponse = {
  id: string;
  name: string;
  folderId: string | null;
  width: number;
  height: number;
  items: CanvasItemPayload[];
  createdAt: number;
  role: "draft" | "filled";
  fields: TemplateFieldPayload[];
};

const mapFolder = (folder: any): FolderResponse => ({
  id: folder._id.toString(),
  name: folder.name,
  parentId: folder.parentId ?? null
});

const mapTemplate = (template: any): TemplateResponse => ({
  id: template._id.toString(),
  name: template.name,
  folderId: template.folderId ?? null,
  width: template.width,
  height: template.height,
  items: template.items ?? [],
  createdAt: new Date(template.createdAt ?? Date.now()).getTime(),
  role: template.role ?? "draft",
  fields: template.fields ?? []
});

const findDescendantFolderIds = async (rootId: string): Promise<string[]> => {
  const ids: string[] = [rootId];
  const queue: string[] = [rootId];

  while (queue.length) {
    const current = queue.shift()!;
    const children = await FolderModel.find({ parentId: current }).select("_id").lean();
    children.forEach((child) => {
      const childId = child._id.toString();
      ids.push(childId);
      queue.push(childId);
    });
  }

  return ids;
};

const placeholderRegex = /\{\{([A-Za-z0-9_]+)\}\}/g;

const extractFieldsFromItems = (items: CanvasItemPayload[]): TemplateFieldPayload[] => {
  const map = new Map<string, TemplateFieldPayload>();
  items.forEach((item) => {
    if (item.type !== "TEXT" || !item.content) {
      return;
    }
    const matches = item.content.matchAll(new RegExp(placeholderRegex));
    for (const match of matches) {
      const name = match[1];
      if (!map.has(name)) {
        map.set(name, { name, label: name, targetItemId: item.id });
      }
    }
  });
  return Array.from(map.values());
};

const replacePlaceholders = (value: string, row: MergeRow): string => {
  if (!value) return value;
  return value.replace(placeholderRegex, (_match, key) => {
    const replacement = row[key];
    return typeof replacement === "string" ? replacement : "";
  });
};

const cloneItemsWithData = (items: CanvasItemPayload[], row: MergeRow): CanvasItemPayload[] =>
  items.map((item) => {
    if (item.type !== "TEXT") {
      return { ...item };
    }
    return {
      ...item,
      content: replacePlaceholders(item.content, row)
    };
  });

const buildPrompt = (context: string, productType: string) => `
Bạn là một chuyên gia copywriting cho bao bì sản phẩm.
Hãy viết một câu slogan hoặc nội dung ngắn gọn, hấp dẫn để in lên tem nhãn cho sản phẩm: "${productType}".
Ngữ cảnh/Ghi chú thêm: "${context}".

Yêu cầu:
- Ngắn gọn (dưới 15 từ).
- Bắt mắt, thu hút.
- Chỉ trả về nội dung văn bản, không có dấu ngoặc kép hay giải thích.
`;

const generateLabelContent = async (context: string, productType: string) => {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY is missing");
  }

  const ai = new GoogleGenAI({ apiKey });
  const prompt = buildPrompt(context, productType);
  const response = await ai.models.generateContent({
    model: GEMINI_MODEL,
    contents: prompt
  });

  const text = response.text?.trim();
  if (!text) {
    throw new Error("Gemini response was empty");
  }

  return text;
};

app.get("/health", (_req: Request, res: Response) => {
  res.json({ ok: true });
});

app.get("/api/folders", async (_req: Request, res: Response) => {
  try {
    const folders = await FolderModel.find().sort({ createdAt: 1 }).lean();
    res.json(folders.map(mapFolder));
  } catch (error) {
    console.error("Failed to list folders", error);
    res.status(500).json({ message: "Không thể tải thư mục" });
  }
});

app.post("/api/folders", async (req: Request, res: Response) => {
  const { name, parentId } = req.body ?? {};

  if (!name) {
    return res.status(400).json({ message: "Tên thư mục là bắt buộc" });
  }

  try {
    const folder = await FolderModel.create({ name, parentId: parentId ?? null });
    res.status(201).json(mapFolder(folder));
  } catch (error) {
    console.error("Failed to create folder", error);
    res.status(500).json({ message: "Không thể tạo thư mục" });
  }
});

app.delete("/api/folders/:id", async (req: Request, res: Response) => {
  const { id } = req.params;

  try {
    const idsToDelete = await findDescendantFolderIds(id);
    await FolderModel.deleteMany({ _id: { $in: idsToDelete } });
    const templates = await TemplateModel.find({ folderId: { $in: idsToDelete } }).select("_id").lean();
    const templateIds = templates.map((tpl) => tpl._id.toString());
    if (templateIds.length) {
      await TemplateModel.deleteMany({ _id: { $in: templateIds } });
    }
    res.json({ deletedFolderIds: idsToDelete, deletedTemplateIds: templateIds });
  } catch (error) {
    console.error("Failed to delete folder", error);
    res.status(500).json({ message: "Không thể xóa thư mục" });
  }
});

app.get("/api/templates", async (_req: Request, res: Response) => {
  try {
    const templates = await TemplateModel.find().sort({ createdAt: -1 }).lean();
    res.json(templates.map(mapTemplate));
  } catch (error) {
    console.error("Failed to list templates", error);
    res.status(500).json({ message: "Không thể tải mẫu" });
  }
});

const validateTemplatePayload = (payload: TemplatePayload): payload is TemplateCreatePayload => {
  return (
    typeof payload.name === "string" &&
    typeof payload.width === "number" &&
    typeof payload.height === "number" &&
    Array.isArray(payload.items) &&
    "folderId" in payload
  );
};

app.post("/api/templates", async (req: Request<unknown, unknown, TemplatePayload>, res: Response) => {
  const payload = { ...req.body, folderId: (req.body?.folderId ?? null) } as TemplatePayload;

  if (!validateTemplatePayload(payload)) {
    return res.status(400).json({ message: "Thiếu dữ liệu mẫu" });
  }

  const { name, folderId, width, height, items } = payload;
  const extractedFields = extractFieldsFromItems(items);

  try {
    const existing = await TemplateModel.findOne({ name, folderId: folderId ?? null }).exec();

    if (existing) {
      const nextFields = extractedFields.length ? extractedFields : existing.fields;
      existing.set({ width, height, items, fields: nextFields });
      await existing.save();
      return res.json(mapTemplate(existing));
    }

    const template = await TemplateModel.create({
      name,
      folderId: folderId ?? null,
      width,
      height,
      items,
      fields: extractedFields,
      role: "draft"
    });
    res.status(201).json(mapTemplate(template));
  } catch (error) {
    console.error("Failed to create template", error);
    res.status(500).json({ message: "Không thể lưu mẫu" });
  }
});

app.put("/api/templates/:id", async (req: Request<{ id: string }, unknown, TemplatePayload>, res: Response) => {
  const { id } = req.params;
  const payload = { ...req.body, folderId: (req.body?.folderId ?? null) } as TemplatePayload;

  if (!validateTemplatePayload(payload)) {
    return res.status(400).json({ message: "Thiếu dữ liệu mẫu" });
  }

  try {
    const extractedFields = payload.items ? extractFieldsFromItems(payload.items) : [];
    const updated = await TemplateModel.findByIdAndUpdate(
      id,
      {
        name: payload.name,
        folderId: payload.folderId ?? null,
        width: payload.width,
        height: payload.height,
        items: payload.items,
        ...(extractedFields.length ? { fields: extractedFields } : {})
      },
      { new: true }
    ).lean();

    if (!updated) {
      return res.status(404).json({ message: "Không tìm thấy mẫu" });
    }

    res.json(mapTemplate(updated));
  } catch (error) {
    console.error("Failed to update template", error);
    res.status(500).json({ message: "Không thể cập nhật mẫu" });
  }
});

app.delete("/api/templates/:id", async (req: Request<{ id: string }>, res: Response) => {
  const { id } = req.params;
  try {
    await TemplateModel.findByIdAndDelete(id);
    res.status(204).send();
  } catch (error) {
    console.error("Failed to delete template", error);
    res.status(500).json({ message: "Không thể xóa mẫu" });
  }
});

app.post("/api/templates/:id/fields/refresh", async (req: Request<{ id: string }>, res: Response) => {
  const { id } = req.params;
  try {
    const template = await TemplateModel.findById(id);
    if (!template) {
      return res.status(404).json({ message: "Không tìm thấy mẫu" });
    }
    const fields = extractFieldsFromItems((template.items ?? []) as CanvasItemPayload[]);
    template.set({ fields });
    await template.save();
    res.json(mapTemplate(template));
  } catch (error) {
    console.error("Failed to refresh fields", error);
    res.status(500).json({ message: "Không thể làm mới trường" });
  }
});

app.post("/api/merge", async (req: Request, res: Response) => {
  const { templateId, folderId, rows } = req.body ?? {};

  if (!templateId || !Array.isArray(rows) || rows.length === 0) {
    return res.status(400).json({ message: "Thiếu dữ liệu merge" });
  }

  try {
    const template = await TemplateModel.findById(templateId).lean();
    if (!template) {
      return res.status(404).json({ message: "Không tìm thấy phôi" });
    }

    const effectiveFields = (template.fields && template.fields.length)
      ? template.fields
      : extractFieldsFromItems((template.items ?? []) as CanvasItemPayload[]);

    const createdTemplates: TemplateResponse[] = [];

    for (let index = 0; index < rows.length; index += 1) {
      const row = rows[index] as MergeRow;
      const targetFolderId = folderId ?? template.folderId ?? null;
      const clonedItems = cloneItemsWithData((template.items ?? []) as CanvasItemPayload[], row);
      const preferredNameField = row.__name || row.name || (effectiveFields[0]?.name ? row[effectiveFields[0].name] : undefined);
      const generatedName = (preferredNameField && preferredNameField.trim().length)
        ? preferredNameField.trim()
        : `${template.name} #${index + 1}`;

      const newTemplate = await TemplateModel.create({
        name: generatedName,
        folderId: targetFolderId,
        width: template.width,
        height: template.height,
        items: clonedItems,
        fields: effectiveFields,
        role: "filled"
      });

      createdTemplates.push(mapTemplate(newTemplate));
    }

    res.status(201).json(createdTemplates);
  } catch (error) {
    console.error("Failed to merge template", error);
    res.status(500).json({ message: "Không thể merge phôi" });
  }
});

app.get("/api/labels", async (_req: Request, res: Response) => {
  try {
    const labels = await LabelModel.find().sort({ createdAt: -1 }).limit(50).lean();
    res.json(labels);
  } catch (error) {
    console.error("Failed to list labels", error);
    res.status(500).json({ message: "Không thể tải danh sách" });
  }
});

app.post("/api/labels", async (req: Request, res: Response) => {
  const { context, productType } = req.body ?? {};

  if (!context || !productType) {
    return res.status(400).json({ message: "context và productType là bắt buộc" });
  }

  try {
    const text = await generateLabelContent(context, productType);
    const label = await LabelModel.create({ context, productType, text });
    res.status(201).json(label);
  } catch (error) {
    console.error("Failed to generate label", error);
    res.status(500).json({ message: "Không thể tạo nội dung" });
  }
});

let serverStarted = false;

const start = async () => {
  try {
    if (mongoose.connection.readyState === 0) {
      await mongoose.connect(MONGO_URL);
    }

    if (!serverStarted && process.env.VERCEL !== "1") {
      app.listen(PORT, () => {
        console.log(`API server listening on http://localhost:${PORT}`);
      });
      serverStarted = true;
    }
  } catch (error) {
    console.error("Unable to start server", error);
    process.exit(1);
  }
};

start();

export default app;
