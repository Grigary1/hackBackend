import express from 'express';
import registerUser, { adminLogin, loginUser, sendOtp, verifyOtp } from '../controllers/userController.js';



const userRouter=express.Router();

userRouter.post('/send-otp',sendOtp);
userRouter.post('/verify-otp',verifyOtp);
userRouter.post('/register',registerUser);
userRouter.post('/login',loginUser);
userRouter.post('/admin',adminLogin);


export default userRouter;