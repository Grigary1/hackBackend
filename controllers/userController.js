import userModel from "../models/userModel.js";
import bcrypt, { hash } from 'bcrypt';
import jwt from 'jsonwebtoken'
import validator from 'validator'
import isEmail from "validator/lib/isEmail.js";
import nodemailer from "nodemailer";
import e from "express";
const createToken = (id) => {
    return jwt.sign({ id }, process.env.JWT_SECRET)
}

const otps = new Map();

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
            from: '"G-Cart" grigusss23@gmail.com',
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
            return res.status(404).json({ success: false, message: "User does not exists" ,});
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
        console.log("Error : ",error.message);
        return res.json({ success: false, message: error.message });
    }

};

const registerUser = async (req, res) => {
    try {
        const { email, password, name } = req.body;
        const exists = await userModel.findOne({ email });
        if (exists) {
            return res.json({ success: false, message: "User already exists" });
        }
        //validating
        if (!validator.isEmail(email)) {
            return res.json({ success: false, message: "Enter valid email" });
        }
        if (password.length < 6) {
            return res.json({ success: false, message: "Enter strong password" });
        }
        const hashedPassword = await bcrypt.hash(password, 10);
        const user = new userModel({
            name, email, password: hashedPassword,
        })
        await user.save();
        return res.json({ success: true, message: "User added" })

    } catch (error) {
        return res.json({ success: false, message: error });
    }
};

export default registerUser;



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