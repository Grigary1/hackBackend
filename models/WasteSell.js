// models/WasteSell.js

import mongoose from "mongoose";

const WasteSellSchema = new mongoose.Schema({
  sellerName: {
    type: String,
    required: true,
  },
  quantity: {
    type: Number,
    required: true,
    min: [1, "Minimum quantity must be at least 1 kg"],
  },
  location: {
    type: String,
    required: true,
  },
  pricePerKg: {
    type: Number,
    required: true,
  },
  phoneNumber: {
    type: String,
    required: true,
  },
  landmark: {
    type: String,
    default: "",
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
});

export default mongoose.model("WasteSell", WasteSellSchema);
