// models/DisposalCenter.js
import mongoose from "mongoose";

const disposalCenterSchema = new mongoose.Schema({
  name: String,
  type: String,
  location: {
    type: {
      type: String,
      enum: ["Point"],
      default: "Point",
    },
    coordinates: {
      type: [Number], // [longitude, latitude]
      required: true,
    },
  },
});

disposalCenterSchema.index({ location: "2dsphere" }); // Enable geo indexing

const DisposalCenter = mongoose.model("DisposalCenter", disposalCenterSchema);
export default DisposalCenter;
