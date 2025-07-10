import userModel from "../models/userModel.js";
import bcrypt, { hash } from 'bcrypt';
import jwt from 'jsonwebtoken'
import validator from 'validator'
import isEmail from "validator/lib/isEmail.js";
import FormData from 'form-data';
import fs from 'fs';
import axios from 'axios'

import nodemailer from "nodemailer";
import e from "express";
import DisposalCenter from "../models/DisposalCenter.js";


import path from "path";
import { spawn } from "child_process";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const createToken = (id) => {
    return jwt.sign({ id }, process.env.JWT_SECRET)
}

const otps = new Map();

export const checkMe=async(req,res)=>{
    try {
        return res.status(200).json({
            success:true,message:"working"
        })
    } catch (error) {
        
    }
}

export const addBin = async (req, res) => {
    console.log("bin")
    try {
      const { latitude, longitude, type, name } = req.body;
  
      if (!latitude || !longitude) {
        return res.status(400).json({ success: false, message: "Missing coordinates" });
      }
  
      const bin = new DisposalCenter({
        name: name || "Unnamed Bin",
        type: type || "general",
        location: {
          type: "Point",
          coordinates: [longitude, latitude], // GeoJSON format [lng, lat]
        },
      });
  
      await bin.save();
  
      res.status(201).json({ success: true, message: "Bin location saved", data: bin });
    } catch (error) {
      console.error("Add Bin Error:", error);
      res.status(500).json({ success: false, message: "Server error" });
    }
  };
  


export const disposeWaste = async (req, res) => {
    console.log("Dispose")
    try {
      const { latitude, longitude } = req.body;
  
      if (!latitude || !longitude) {
        return res.status(400).json({ success: false, message: "Coordinates required" });
      }
  
      const userLocation = {
        type: "Point",
        coordinates: [longitude, latitude], // GeoJSON expects [lng, lat]
      };
  
      const centers = await DisposalCenter.find({
        location: {
          $near: {
            $geometry: userLocation,
            $maxDistance: 20, // 20 meters radius
          },
        },
      });
  
      return res.status(200).json({ success: true, data: centers });
    } catch (error) {
      console.error("Geolocation query error:", error);
      return res.status(500).json({ success: false, message: error.message });
    }
  };


  export const readImage = async (req, res) => {
    const image = req.file;
  
    if (!image) {
      return res.status(400).json({ error: "No image uploaded" });
    }
  
    const pythonScript = path.join(__dirname, "..", "py-model-server", "predict.py");
    const py = spawn("python3", [pythonScript, image.path]);
  
    let result = "";
    let error = "";
  
    py.stdout.on("data", (data) => {
      result += data.toString();
    });
  
    py.stderr.on("data", (data) => {
      error += data.toString();
    });
  
    py.on("close", (code) => {
      // Delete uploaded file after prediction
      fs.unlinkSync(image.path);
  
      if (code !== 0 || error) {
        return res.status(500).json({ error: error || "Prediction failed" });
      }
  
      try {
        const output = JSON.parse(result);
        res.json(output);
      } catch (err) {
        res.status(500).json({ error: "Invalid response from Python script" });
      }
    });
  };

export const verifyOtp = async (req, res) => {
    const { email, otp } = req.body;
    const record = otps.get(email);

    if (!record) return res.status(400).json({ message: "OTP not found" });

    const { otp: savedOtp, expiresAt } = record;

    if (Date.now() > expiresAt) {
        otps.delete(email);
        return res.status(400).json({ message: "OTP expired" });
    }

    if (otp !== savedOtp) {
        return res.status(400).json({ message: "Invalid OTP" });
    }

    otps.delete(email);
    res.json({ message: "OTP verified successfully" });
};


export const sendOtp = async (req, res) => {
    try {
        const { email } = req.body;
        if (!isEmail(email)) {
            return res.status(400).json({
                success: false,
                message: "Invalid email"
            })
        }
        const exists = await userModel.findOne({ email });
        if (exists) {
            return res.status(400).json({
                success: false,
                message: "Email already registered. Please try to login"
            })
        }
        const otp = Math.floor(100000 + Math.random() * 900000).toString();
        otps.set(email, { otp, expiresAt: Date.now() + 5 * 60 * 1000 });
        const transporter = nodemailer.createTransport({
            service: 'gmail',
            auth: {
                user: 'grigusss23@gmail.com',
                pass: 'xfqm adyn sxey vbxq'
            }
        });

        // Email content
        const mailOptions = {
            from: '"DropIt" grigusss23@gmail.com',
            to: email,
            subject: 'Your OTP Code',
            text: `Your OTP code is: ${otp}. It is valid for 5 minutes.`,
            html: `<p>Your OTP code is: <b>${otp}</b>. It is valid for 5 minutes.</p>`
        };

        try {
            await transporter.sendMail(mailOptions);
            res.json({ message: "OTP sent successfully" });
        } catch (err) {
            console.error(err);
            res.status(500).json({ message: "Failed to send OTP" });
        }
    } catch (error) {
        console.error(error);
        return res.status(201).json({
            success: false,
            message: error.message
        })
    }
}

//Route for user login
const loginUser = async (req, res) => {
    console.log("loginUser");
    try {
        const { email, password } = req.body;
        const data = await userModel.findOne({ email });
        if (!data) {
            return res.status(404).json({ success: false, message: "User does not exists", });
        }
        const storedPassword = data.password;
        const match = await bcrypt.compare(password, storedPassword);
        if (!match) {
            return res.status(404).json({ success: false, message: "Incorrect password" });
        }
        const token = jwt.sign(
            { id: data._id, email: data.email },
            process.env.JWT_SECRET,
            { expiresIn: "10h" }
        );
        return res.json({
            success: true,
            message: "Login successful",
            token,
            user: { id: data._id, name: data.name, email: data.email },
        });
    } catch (error) {
        console.log("Error : ", error.message);
        return res.json({ success: false, message: error.message });
    }

};


const registerUser = async (req, res) => {
    try {
        const { email, password, name, age, location } = req.body;

        const exists = await userModel.findOne({ email });
        if (exists) {
            return res.json({ success: false, message: "User already exists" });
        }

        // Validations
        if (!validator.isEmail(email)) {
            return res.json({ success: false, message: "Enter a valid email" });
        }
        if (password.length < 6) {
            return res.json({ success: false, message: "Password too short" });
        }
        if (!age || !location) {
            return res.json({ success: false, message: "Age and location are required" });
        }

        const hashedPassword = await bcrypt.hash(password, 10);

        const user = new userModel({
            name,
            email,
            password: hashedPassword,
            age,
            location,
        });

        await user.save();

        return res.json({ success: true, message: "User added" });
    } catch (error) {
        return res.json({ success: false, message: error.message });
    }
};





//Route for admin login
const adminLogin = async (req, res) => {
    try {
        const { email, password } = req.body;
        if (email === process.env.ADMIN_EMAIL && password === process.env.ADMIN_PASSWORD) {
            const token = jwt.sign(email + password, process.env.JWT_SECRET);
            return res.status(200).json({ success: true, token });
        }
        else {
            return res.json({ success: false, message: "Invalid credentials" });
        }
    } catch (error) {
        return res.send("False");
    }
}
export { loginUser, registerUser, adminLogin };